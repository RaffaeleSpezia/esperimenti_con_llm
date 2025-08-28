#!/usr/bin/env python3
# Mini Chat con cronologia ordinata + stampa contesto + streaming token-per-token
# Target: Qwen2.5-Coder-7B-Instruct su ROCm (ma funziona con qualunque chat model HF compatibile)

import os, sys, time, tempfile, subprocess, signal
from typing import List, Dict

MODEL_DIR = os.environ.get("MODEL_DIR", "/mnt/raid0/qwen2.5-coder-7b-instruct")
DO_SAMPLE = os.environ.get("DO_SAMPLE", "false").lower() in ("1","true","yes")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2" if DO_SAMPLE else "1.0"))
TOP_P = float(os.environ.get("TOP_P", "0.95" if DO_SAMPLE else "1.0"))
TOP_K = int(os.environ.get("TOP_K", "40" if DO_SAMPLE else "0"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))

# ===== Utils =====
def open_in_nano(initial_text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".md", prefix="mini_chat_"); os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(initial_text)
        subprocess.run(["nano", "+999999", "--softwrap", "--tabstospaces", path], check=False)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        try: os.unlink(path)
        except: pass

PROMPT_BEGIN = "<<<SCRIVI QUI IL TUO PROMPT>>>"
PROMPT_END   = "<<<FINE>>>"

def render_editor_buffer(history: List[Dict[str,str]]) -> str:
    out = []
    out.append("# MINI CHAT")
    out.append("# Scrivi il tuo prompt tra le due righe marcate qui sotto e salva/chiudi.")
    out.append("# Comando per uscire: /exit")
    out.append("")
    # Mostra cronologia in ordine
    for i, m in enumerate(history, 1):
        role = m["role"].upper()
        out.append(f"### [{i}] {role}")
        out.append(m["content"])
        out.append("")
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

# ===== App =====
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
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"📍 Device: {model.device}")

    # Streamer per stampa token-per-token
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    # Parametri di generazione minimali e coerenti
    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": True,
        "do_sample": DO_SAMPLE,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.eos_token_id,
    }
    if DO_SAMPLE:
        gen_kwargs["temperature"] = TEMPERATURE
        gen_kwargs["top_p"] = TOP_P
        if TOP_K > 0:
            gen_kwargs["top_k"] = TOP_K

    history: List[Dict[str,str]] = []  # lista di dict: {"role": "user"/"assistant", "content": str}

    print("\n📝 Apro nano. (Scrivi /exit per terminare)")
    while True:
        edited = open_in_nano(render_editor_buffer(history))
        user_prompt = extract_user_prompt(edited)

        if user_prompt.strip().lower() == "/exit":
            print("👋 Uscita.")
            break
        if not user_prompt.strip():
            # nessun input → ricomincia
            continue

        # Aggiungi turno utente alla cronologia
        history.append({"role":"user", "content": user_prompt})

        # Costruisci contesto completo come lista messaggi
        msgs = []
        for m in history:
            msgs.append({"role": m["role"], "content": m["content"]})

        # Serializza nel formato del modello (ChatML per Qwen)
        chat_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        # Stampa il contesto esatto che passeremo al modello
        print("\n====== CONTENSTO COMPLETO PASSATO AL MODELLO ======")
        print(chat_str)
        print("===================================================\n")

        # Tokenizza + genera con streaming
        inputs = tok(chat_str, return_tensors="pt").to(model.device)

        # Decodificheremo solo i token generati (niente eco del prompt)
        input_len = inputs["input_ids"].shape[1]

        print("🤖 Generazione (streaming attivo)...\n")
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                **gen_kwargs,
                streamer=streamer
            )

        # out contiene input + output; estrai solo la parte generata
        seq = out[0]
        gen_only = seq[input_len:]
        assistant_text = tok.decode(gen_only, skip_special_tokens=True).strip()

        # Append alla history
        history.append({"role":"assistant", "content": assistant_text})

        print("\n✅ Turno completato.\n")

    # Salvataggio transcript semplice
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"mini_chat_{ts}.md"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"# Mini Chat — {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**MODEL_DIR:** {MODEL_DIR}\n\n")
        for i, m in enumerate(history, 1):
            f.write(f"### [{i}] {m['role'].upper()}\n{m['content']}\n\n")
    print(f"💾 Transcript salvato: {fname}")

if __name__ == "__main__":
    main()
