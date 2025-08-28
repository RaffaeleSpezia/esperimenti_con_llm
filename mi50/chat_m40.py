#!/usr/bin/env python3
# Mini Chat per Tesla M40 con Gemma3-4B-IT
# - Tag chat Gemma3: <start_of_turn>user|model ... <end_of_turn>
# - Stampa contesto completo passato al modello
# - Streaming token-per-token
# - Stop su "<end_of_turn>"
# - Niente system prompt / artifacts

import os, sys, time, tempfile, subprocess, signal
from typing import List, Dict

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/lele/clara/clara-gemma2/gemma3-4b-it")

DO_SAMPLE       = os.environ.get("DO_SAMPLE", "false").lower() in ("1","true","yes")
TEMPERATURE     = float(os.environ.get("TEMPERATURE", "0.2" if DO_SAMPLE else "1.0"))
TOP_P           = float(os.environ.get("TOP_P", "0.95" if DO_SAMPLE else "1.0"))
TOP_K           = int(os.environ.get("TOP_K", "40" if DO_SAMPLE else "0"))
MAX_NEW_TOKENS  = int(os.environ.get("MAX_NEW_TOKENS", "1024"))

PROMPT_BEGIN = "<<<SCRIVI QUI IL TUO PROMPT>>>"
PROMPT_END   = "<<<FINE>>>"

def open_in_nano(initial_text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".md", prefix="mini_chat_m40_"); os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(initial_text)
        subprocess.run(["nano", "+999999", "--softwrap", "--tabstospaces", path], check=False)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        try: os.unlink(path)
        except: pass

def render_editor_buffer(history: List[Dict[str,str]]) -> str:
    out = []
    out.append("# MINI CHAT (M40 + Gemma3-4B-IT)")
    out.append("# Scrivi il tuo prompt tra le due righe e salva/chiudi.")
    out.append("# /exit per uscire.")
    out.append("")
    for i, m in enumerate(history, 1):
        out.append(f"### [{i}] {m['role'].upper()}")
        out.append(m["content"]); out.append("")
    out.append("----")
    out.append(PROMPT_BEGIN)
    out.append("")
    out.append(PROMPT_END)
    out.append("")
    return "\n".join(out)

def extract_user_prompt(edited: str) -> str:
    try:
        start = edited.index(PROMPT_BEGIN) + len(PROMPT_BEGIN)
        end   = edited.index(PROMPT_END, start)
        return edited[start:end].strip()
    except ValueError:
        return ""

# ===== Manuale: serializza la cronologia con i tag Gemma3 =====
def format_gemma3_chat(history: List[Dict[str,str]], next_user: str) -> str:
    parts = []
    for m in history:
        if m["role"] == "user":
            parts.append("<start_of_turn>user\n"  + m["content"].rstrip() + "\n<end_of_turn>\n")
        else:  # "model"
            parts.append("<start_of_turn>model\n" + m["content"].rstrip() + "\n<end_of_turn>\n")
    # aggiungi il prossimo turno utente + prompt di generazione per model
    parts.append("<start_of_turn>user\n"  + next_user.rstrip() + "\n<end_of_turn>\n")
    parts.append("<start_of_turn>model\n")  # il modello deve continuare da qui e chiudere con <end_of_turn>
    return "".join(parts)

# ===== Stopping su "<end_of_turn>" =====
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        # codifica tokenizzata della stringa di stop
        self.stop_ids = tokenizer.encode("<end_of_turn>", add_special_tokens=False)
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if not self.stop_ids:
            return False
        seq = input_ids[0].tolist()
        L = len(self.stop_ids)
        return len(seq) >= L and seq[-L:] == self.stop_ids

def main():
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
    import torch

    print(f"✅ Carico modello: {MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype="auto",
        device_map="auto",          # su M40 (12GB) userà offload/CPU se serve
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager" # più compatibile con M40
    )
    print(f"📍 Device: {model.device}")

    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": True,
        "do_sample": DO_SAMPLE,
        "eos_token_id": tok.eos_token_id,   # non ci affidiamo a questo; usiamo stopping esplicito
        "pad_token_id": tok.eos_token_id,
    }
    if DO_SAMPLE:
        gen_kwargs["temperature"] = TEMPERATURE
        gen_kwargs["top_p"] = TOP_P
        if TOP_K > 0:
            gen_kwargs["top_k"] = TOP_K

    stop_list = StoppingCriteriaList([StopOnEndOfTurn(tok)])

    # history come coppie ordinate: role in {"user","model"}
    history: List[Dict[str,str]] = []

    print("\n📝 Apro nano. (Scrivi /exit per terminare)")
    while True:
        edited = open_in_nano(render_editor_buffer(history))
        user_prompt = extract_user_prompt(edited)

        if user_prompt.strip().lower() == "/exit":
            print("👋 Uscita.")
            break
        if not user_prompt.strip():
            continue

        # aggiungi il turno utente alla cronologia
        history.append({"role":"user","content":user_prompt})

        # serializza nel formato Gemma3 (manuale, con tag)
        chat_str = format_gemma3_chat(history[:-1] if history and history[-1]["role"]=="user" else history, user_prompt)

        # mostra esattamente cosa inviamo
        print("\n====== CONTESTO COMPLETO PASSATO AL MODELLO ======")
        print(chat_str)
        print("==================================================\n")

        inputs = tok(chat_str, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        print("🤖 Generazione (streaming attivo, stop su <end_of_turn>)...\n")
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                **gen_kwargs,
                streamer=streamer,
                stopping_criteria=stop_list
            )

        # estrai solo la parte generata
        seq = out[0]
        gen_only = seq[input_len:]
        text = tok.decode(gen_only, skip_special_tokens=False)  # NON skip, così vediamo <end_of_turn>
        # taglia in output finale al primo <end_of_turn>
        end_tag = "<end_of_turn>"
        cut = text.find(end_tag)
        assistant_text = text[:cut].strip() if cut != -1 else text.strip()

        # aggiungi il turno model alla cronologia
        history.append({"role":"model","content":assistant_text})
        print("\n✅ Turno completato.\n")

    # salva transcript
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"mini_chat_m40_gemma3_{ts}.md"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"# Mini Chat M40 Gemma3 — {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**MODEL_DIR:** {MODEL_DIR}\n\n")
        for i, m in enumerate(history, 1):
            f.write(f"### [{i}] {m['role'].upper()}\n{m['content']}\n\n")
    print(f"💾 Transcript salvato: {fname}")

if __name__ == "__main__":
    main()
