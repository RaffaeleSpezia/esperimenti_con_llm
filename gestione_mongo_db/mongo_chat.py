#!/usr/bin/env python3
# mongo_chat_ortociccio.py
# Chat con editing + esecuzione iterativa delle query Mongo.
# Supporta <mongo_instruction>…</mongo_instruction> e blocchi ```mongosh …``` / ```mongo …```.
# Esegue mongosh su FILE TEMPORANEO, logga CMD/FILE/JS/STDOUT/STDERR/RC e inietta risposte reali.

import os, sys, time, tempfile, subprocess, signal, re
from typing import List, Dict, Tuple

# =========================
# CONFIG
# =========================
MODEL_DIR = os.environ.get("MODEL_DIR", "/mnt/raid0/qwen2.5-coder-7b-instruct")
DO_SAMPLE = os.environ.get("DO_SAMPLE", "false").lower() in ("1","true","yes")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2" if DO_SAMPLE else "1.0"))
TOP_P = float(os.environ.get("TOP_P", "0.95" if DO_SAMPLE else "1.0"))
TOP_K = int(os.environ.get("TOP_K", "40" if DO_SAMPLE else "0"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
MAX_CHAIN_STEPS = int(os.environ.get("MAX_CHAIN_STEPS", "6"))  # safety

# Mongo
MONGO_URI     = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "ortociccio")
MONGOSH_BIN   = os.environ.get("MONGOSH_BIN", "mongosh")  # es. "/usr/bin/mongosh"
# Vuoto di default per massima visibilità; se vuoi, esporta MONGOSH_ARGS="--quiet"
MONGOSH_ARGS  = os.environ.get("MONGOSH_ARGS", "").split() if os.environ.get("MONGOSH_ARGS") else []

# Tag Mongo
TAG_OPEN   = "<mongo_instruction>"
TAG_CLOSE  = "</mongo_instruction>"
RESP_OPEN  = "<mongo_response>"
RESP_CLOSE = "</mongo_response>"

# Delimitatori input utente
PROMPT_BEGIN = "<<<SCRIVI QUI IL TUO PROMPT>>>"
PROMPT_END   = "<<<FINE>>>"

# =========================
# PROMPT LIZA (rigoroso)
# =========================
LIZA_SYSTEM = """Ciao, il tuo nome è Liza.
Sei un assistente che interagisce con un database MongoDB e con un utente umano.

## Regole fondamentali
- NON INVENTARE MAI DATI. Se un’informazione non è nel database o nel messaggio dell’utente, dillo.
- NON SCRIVERE MAI tu i blocchi `<mongo_response>` né il loro contenuto. Sono inseriti solo dal sistema dopo l’esecuzione reale delle query.
- Se servono dati dal DB, nel tuo messaggio scrivi SOLTANTO i comandi in uno dei due formati:
  1) Tag:
     <mongo_instruction>
     … comandi MongoDB …
     </mongo_instruction>
  2) Blocco fenced:
     ```mongosh
     … comandi MongoDB …
     ```
  e aggiungi la riga: “Eseguo la/le query e attendo la risposta del database.”
- Puoi mettere più istruzioni una sotto l’altra; verranno eseguite in ordine e riceverai l’output di ciascuna.
- NON trarre conclusioni prima di aver ricevuto un `<mongo_response>`.
- Quando ricevi un `<mongo_response>`, rispondi basandoti solo su quei dati; se sono insufficienti, proponi la prossima query.
- Se un campo richiesto non esiste nello schema, dillo e proponi un’estensione o una query alternativa.

## Database `ortociccio`
- units: unit_id, location{name,lat,lon}, tags[], created_at
- sensor_data: unit_id, sensor_id (es. h1_u1, t3_u2), sensor_type ("soil_humidity" | "soil_temp"), value, unit ("%" | "°C"), ts
- control_events: unit_id, event ("pump_on" | "pump_off" | "tank_level" | …), payload{…}, ts
- weather_data: unit_id, temperature, humidity, wind_speed, ts

## Linee guida query
- “Ultimo”: ordina per ts desc e limita a 1.
- Conteggi: countDocuments() o $group con $sum.
- Proietta solo i campi necessari. Operazioni di scrittura solo su richiesta esplicita dell’utente.

## Formato messaggi
- Prima di `<mongo_response>`: SOLO istruzioni (tag o ```mongosh```).
- Dopo `<mongo_response>`: spiega e, se serve, proponi un’ulteriore istruzione.
"""

# =========================
# UTILS: editor + parsing
# =========================
SECTION_RE = re.compile(r"^### \[(\d+)\] (SYSTEM|USER|ASSISTANT)\s*$", flags=re.MULTILINE)

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

def render_editor_buffer(history: List[Dict[str,str]]) -> str:
    out = []
    out.append("# MINI CHAT — Modifica liberamente TUTTO il contesto")
    out.append("# Blocchi ammessi: ### [n] SYSTEM | USER | ASSISTANT")
    out.append("# Il nuovo prompt utente va tra i delimitatori in fondo; /exit per uscire.")
    out.append("")
    for i, m in enumerate(history, 1):
        out.append(f"### [{i}] {m['role'].upper()}")
        out.append(m["content"])
        out.append("")
    out.append("----")
    out.append(PROMPT_BEGIN)
    out.append("")
    out.append(PROMPT_END)
    out.append("")
    return "\n".join(out)

def parse_editor_buffer(edited: str) -> Tuple[List[Dict[str,str]], str]:
    try:
        start = edited.index(PROMPT_BEGIN) + len(PROMPT_BEGIN)
        end   = edited.index(PROMPT_END, start)
        user_prompt = edited[start:end].strip()
    except ValueError:
        user_prompt = ""

    main_text = edited
    if PROMPT_BEGIN in edited:
        main_text = edited[:edited.index(PROMPT_BEGIN)]

    history: List[Dict[str,str]] = []
    matches = list(SECTION_RE.finditer(main_text))
    if matches:
        for idx, m in enumerate(matches):
            role = m.group(2).lower()
            s = m.end()
            e = matches[idx+1].start() if idx+1 < len(matches) else len(main_text)
            content = main_text[s:e].strip()
            history.append({"role": role, "content": content})
    else:
        system_text = main_text.strip()
        if system_text:
            history = [{"role":"system", "content": system_text}]

    return history, user_prompt

# =========================
# MONGO BRIDGE (tag + fenced) — parsing
# =========================
MONGO_BLOCK_RE = re.compile(re.escape(TAG_OPEN) + r"(.*?)" + re.escape(TAG_CLOSE), flags=re.DOTALL | re.IGNORECASE)
RESP_BLOCK_RE  = re.compile(re.escape(RESP_OPEN) + r".*?" + re.escape(RESP_CLOSE), flags=re.DOTALL | re.IGNORECASE)
MONGO_FENCE_RE = re.compile(r"```(?:\s*(?:mongosh|mongo)\s*)(.*?)```", flags=re.DOTALL | re.IGNORECASE)

# =========================
# MONGO BRIDGE — helpers per esecuzione
# =========================
def _split_statements(js_code: str) -> list:
    """
    Divide il codice in 'statement' top-level senza rompere stringhe/backtick o blocchi { } ( ) [ ].
    Taglia solo su ';' a profondità 0 e fuori dalle stringhe. L'ultimo chunk (anche senza ';') è incluso.
    """
    s = js_code
    out, buf = [], []
    depth_par, depth_brk, depth_brc = 0, 0, 0
    in_str = None  # "'", '"', '`'
    esc = False

    for ch in s:
        buf.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
        else:
            if ch in ("'", '"', "`"):
                in_str = ch
            elif ch == "(":
                depth_par += 1
            elif ch == ")":
                depth_par = max(0, depth_par - 1)
            elif ch == "[":
                depth_brk += 1
            elif ch == "]":
                depth_brk = max(0, depth_brk - 1)
            elif ch == "{":
                depth_brc += 1
            elif ch == "}":
                depth_brc = max(0, depth_brc - 1)
            elif ch == ";" and depth_par == depth_brk == depth_brc == 0:
                out.append("".join(buf).strip())
                buf = []
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return [st for st in (st.strip() for st in out) if st and not st.lstrip().startswith("//")]

def _make_printing_block(statements: list) -> str:
    """
    Per ogni istruzione:
      - se contiene già console.log/print/printjson → la lascio com'è (stamperà da sola);
      - altrimenti:
          * rimuovo ';' finale (se presente),
          * se è find/aggregate senza stampa → aggiungo .toArray(),
          * eseguo e stampo:
              let __expr = (<istruzione>);
              if (__expr && typeof __expr.then === 'function') { __expr = await __expr; }
              if (typeof __expr !== 'undefined') {
                console.log('--- out[i] ---');
                console.log(JSON.stringify(__expr, null, 2));
              }
    """
    blocks = []
    for i, raw in enumerate(statements, start=1):
        st = raw.strip()
        if ("console.log(" in st) or ("printjson(" in st) or ("print(" in st):
            blocks.append(st if st.endswith(";") else st + ";")
            continue

        core = st[:-1].strip() if st.endswith(";") else st

        if (("find(" in core or "aggregate(" in core)
            and ".toArray()" not in core
            and ".pretty()" not in core
            and "printjson(" not in core
            and "console.log(" not in core):
            core = core + ".toArray()"

        blocks.append(
f"""\
let __expr = ({core});
if (__expr && typeof __expr.then === 'function') {{ __expr = await __expr; }}
if (typeof __expr !== 'undefined') {{
  console.log('--- out[{i}] ---');
  console.log(JSON.stringify(__expr, null, 2));
}}"""
        )
    return "\n".join(blocks) + "\n"

def run_mongosh(js_code: str, db_name: str) -> Tuple[str, int]:
    """
    Esegue mongosh su FILE TEMPORANEO e ritorna (stdout+stderr, rc).
    - Connessione esplicita (Mongo(MONGO_URI) → getDB(db_name))
    - Spezza in statement top-level e stampa l'OUTPUT di ognuno (in ordine)
    - Usa IIFE async per await delle Promise
    - Logga CMD / FILE / JS / STDOUT / STDERR / RC
    """
    raw = js_code.strip()
    if not raw:
        msg = "ERRORE: blocco istruzioni vuoto."
        print("\n=== MONGOSH ESECUZIONE ===")
        print("ERRORE:", msg)
        print("==========================\n")
        return msg, 2

    stmts = _split_statements(raw)
    body  = _make_printing_block(stmts)

    prelude = f"""\
const conn = new Mongo("{MONGO_URI}");
const db = conn.getDB("{db_name}");
"""

    js_final = prelude + "(async () => {\n" + body + "})().catch(e => { console.error(e?.stack || e); process.exit(1); });\n"

    fd, path = tempfile.mkstemp(suffix=".mongo.js", prefix="liza_")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(js_final)

        cmd = [MONGOSH_BIN, *MONGOSH_ARGS, path]

        print("\n=== MONGOSH ESECUZIONE ===")
        print("CMD:", " ".join(cmd))
        print("FILE:", path)
        print("--- JS FILE CONTENT ---")
        print(js_final)
        print("-----------------------")

        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        out_txt = proc.stdout.strip()
        err = proc.stderr.strip()

        print("--- STDOUT ---")
        print(out_txt if out_txt else "(vuoto)")
        print("--- STDERR ---")
        print(err if err else "(vuoto)")
        print("RC:", proc.returncode)
        print("==========================\n")

        if err:
            out_txt = (out_txt + ("\n" if out_txt else "") + "[stderr]\n" + err).strip()

        if not out_txt and proc.returncode == 0:
            out_txt = "(nessun output — nessuna istruzione ha prodotto un valore visualizzabile)"

        return out_txt if out_txt else "(nessun output)", proc.returncode

    except FileNotFoundError:
        msg = f"ERRORE: '{MONGOSH_BIN}' non trovato nel PATH. Imposta MONGOSH_BIN o installa mongosh."
        print("--- EXCEPTION ---")
        print(msg)
        print("==========================\n")
        return msg, 127
    except Exception as e:
        msg = f"ERRORE runtime mongosh: {e}"
        print("--- EXCEPTION ---")
        print(msg)
        print("==========================\n")
        return msg, 1
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass

def _exec_and_inject(code: str, db_name: str) -> str:
    """
    Esegue e restituisce SOLO il blocco <mongo_response> con etichetta.
    Il blocco istruzioni originale (tag o fenced) resta visibile sopra.
    """
    code = code.strip()
    output, rc = run_mongosh(code, db_name)
    header = "risposta del database:"
    if rc != 0:
        header += f" (rc={rc})"
    return f"{RESP_OPEN}\n{header}\n{output}\n{RESP_CLOSE}"

def inject_mongo_responses(assistant_text: str, db_name: str) -> str:
    """Rimuove <mongo_response> “inventati”, esegue i blocchi e inietta SOLO output reale."""
    # 1) rimuovi blocchi risposta eventualmente presenti NEL MESSAGGIO CORRENTE
    cleaned = RESP_BLOCK_RE.sub("", assistant_text)

    # 2) esegui ogni blocco <mongo_instruction>
    def _repl_tag(m: re.Match) -> str:
        code = m.group(1)
        resp = _exec_and_inject(code, db_name)
        return f"{TAG_OPEN}\n{code.strip()}\n{TAG_CLOSE}\n{resp}"

    cleaned = MONGO_BLOCK_RE.sub(_repl_tag, cleaned)

    # 3) esegui i blocchi ```mongosh ...``` / ```mongo ...```
    def _repl_fence(m: re.Match) -> str:
        code = m.group(1)
        resp = _exec_and_inject(code, db_name)
        return f"```mongosh\n{code.strip()}\n```\n{resp}"

    cleaned = MONGO_FENCE_RE.sub(_repl_fence, cleaned)
    return cleaned

def has_mongo_instruction_or_fence(text: str) -> bool:
    return bool(MONGO_BLOCK_RE.search(text) or MONGO_FENCE_RE.search(text))

# =========================
# MODEL HELPERS
# =========================
def generate_from_history(tok, model, history: List[Dict[str,str]], streamer, gen_kwargs) -> str:
    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    chat_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print("\n====== CONTESTO COMPLETO PASSATO AL MODELLO ======")
    print(chat_str)
    print("===================================================\n")

    inputs = tok(chat_str, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    print("🤖 Generazione (streaming attivo)...\n")
    import torch
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs, streamer=streamer)
    seq = out[0]
    gen_only = seq[input_len:]
    text = tok.decode(gen_only, skip_special_tokens=True).strip()
    return text

# =========================
# MAIN
# =========================
def main():
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

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

    # History iniziale
    history: List[Dict[str,str]] = [{"role":"system", "content": LIZA_SYSTEM}]

    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
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

    print("\n📝 Apro nano. (Modifica TUTTO liberamente; scrivi /exit tra i delimitatori per uscire)")
    while True:
        edited = open_in_nano(render_editor_buffer(history))
        new_history, user_prompt = parse_editor_buffer(edited)
        if new_history:
            history = new_history

        if user_prompt.strip().lower() == "/exit":
            print("👋 Uscita.")
            break

        if not user_prompt.strip():
            continue

        # === TURNO UTENTE
        history.append({"role":"user", "content": user_prompt})

        # === CATENA: (query → DB → risposta) finché Liza emette istruzioni
        chain_steps = 0
        while True:
            chain_steps += 1
            if chain_steps > MAX_CHAIN_STEPS:
                history.append({"role":"assistant","content":"(Interrotto per sicurezza: troppi passaggi consecutivi. Dimmi come procedere.)"})
                break

            assistant_msg = generate_from_history(tok, model, history, streamer, gen_kwargs)
            # pulizia: rimuovi <mongo_response> eventualmente “inventati” nel messaggio corrente
            assistant_msg = RESP_BLOCK_RE.sub("", assistant_msg).strip()

            if has_mongo_instruction_or_fence(assistant_msg):
                print("\n🔗 Eseguo query MongoDB e inietto <mongo_response> reale…")
                assistant_with_db = inject_mongo_responses(assistant_msg, MONGO_DB_NAME)
                print("✅ Risposte MongoDB iniettate.\n")
                history.append({"role":"assistant", "content": assistant_with_db})
                continue
            else:
                # Nessuna query → questa è la risposta finale per l’utente
                history.append({"role":"assistant", "content": assistant_msg})
                break

        print("\n✅ Turno completato. Riapro nano per il prossimo input.\n")

    # Transcript finale
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
