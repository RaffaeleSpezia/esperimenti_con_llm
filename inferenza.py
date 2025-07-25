# gradio_final.py
import gradio as gr
import torch
import time
import json
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./gemma3-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🔄 Caricamento tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("🔄 Caricamento modello...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
print("✅ Modello pronto!")

# ---------- funzioni ----------
def generate(prompt,
             temperature=1.0,
             top_p=0.5,
             top_k=15,
             max_tokens=4096,
             repetition_penalty=1.1,
             seed=42):
    if seed is not None:
        torch.manual_seed(int(seed))

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"][0]

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][len(input_ids):]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    speed = len(gen_ids) / max(time.time() - t0, 1e-5)

    # --- blocco “parola [id]” wrappato ---
    full_ids = torch.cat([input_ids, gen_ids]).tolist()
    chunks_html = []

    for tid in full_ids:
        word = tokenizer.decode([tid], skip_special_tokens=False).replace(" ", "·")
        chunk = (
            f'<span style="display:inline-block;margin:2px 4px;padding:2px 4px;border:1px solid #ccc;'
            f'border-radius:4px;font-family:SF Mono,Consolas,monospace;">'
            f'<b>{word}</b><br><span style="color:#00ffff;">[{tid}]</span></span>'
        )
        chunks_html.append(chunk)

    html_blocks = (
        '<div style="line-height:1.8;word-break:break-word;">' +
        "".join(chunks_html) +
        "</div>"
    )

    return (
        text,
        f"{len(gen_ids)} token generati",
        f"{speed:.1f} tok/s",
        html_blocks
    )

def save(prompt, response, temperature, top_p, top_k, max_tokens, repetition_penalty, seed):
    os.makedirs("saved", exist_ok=True)
    filename = f"saved/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    data = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": response,
        "settings": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
        }
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"💾 Salvato in {filename}"

# ---------- interfaccia ----------
with gr.Blocks(title="💾 Gemma-3 + token-view") as demo:
    gr.Markdown("## 💾 Gemma-3 – generazione + token-view")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(lines=4, placeholder="Inserisci prompt...", label="Prompt")
            seed = gr.Number(value=42, precision=0, label="Seed")
            generate_btn = gr.Button("🚀 Genera", variant="primary")
            save_btn = gr.Button("💾 Salva", variant="secondary")
            save_status = gr.Textbox(label="Stato salvataggio", interactive=False)

        with gr.Column(scale=2):
            response = gr.Textbox(lines=8, label="Risposta")
            info_token = gr.Textbox(label="Info token")
            perf = gr.Textbox(label="Performance")

    with gr.Row():
        temperature = gr.Slider(0.1, 2.0, step=0.1, value=1.0, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, step=0.05, value=0.5, label="Top-p")
        top_k = gr.Slider(1, 100, step=1, value=15, label="Top-k")
        max_tokens = gr.Slider(1, 4096, step=1, value=512, label="Max new tokens")
        repetition_penalty = gr.Slider(1.0, 2.0, step=0.1, value=1.1, label="Repetition penalty")

    # visualizzazione token-testo
    token_view = gr.HTML(label="🔍 Token ↔ Testo (prompt + risposta)")

    # eventi
    generate_btn.click(
        fn=generate,
        inputs=[prompt, temperature, top_p, top_k, max_tokens, repetition_penalty, seed],
        outputs=[response, info_token, perf, token_view],
    )

    save_btn.click(
        fn=save,
        inputs=[prompt, response, temperature, top_p, top_k, max_tokens, repetition_penalty, seed],
        outputs=[save_status],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
