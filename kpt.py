import gradio as gr
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Caricamento modello
model_path = "./gemma3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.eval()

# Grafico 2D PCA classico
def crea_grafico_pca(embedding_tensor, save_path="pca_output.png"):
    emb_np = embedding_tensor.detach().cpu().to(torch.float32).numpy()
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb_np)

    plt.figure(figsize=(6, 5))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c="blue", label="Token")
    plt.title("PCA degli embedding dei token generati")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Grafico 3D evolutivo layer-token
def crea_grafico_3d_evolutivo(input_ids, save_path="embedding_3d.png"):
    input_ids = input_ids.unsqueeze(0).to("cuda")
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, return_dict=True)

    hidden_states = outputs.hidden_states  # (layer+1, 1, seq_len, hidden_size)
    seq_len = input_ids.shape[1]
    token_list = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.viridis(torch.linspace(0, 1, seq_len))

    for i in range(seq_len):
        path = torch.stack([layer[0, i].cpu().to(torch.float32) for layer in hidden_states])
        coords = PCA(n_components=3).fit_transform(path)
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        ax.plot(xs, ys, zs, color=colors[i], label=token_list[i])
        ax.text(xs[-1], ys[-1], zs[-1], token_list[i], fontsize=7)

    ax.set_title("🧠 Evoluzione degli embedding nei layer")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Funzione principale
def genera(prompt, temperature, top_k, top_p, penalty, seed, deterministico, max_token):
    if deterministico:
        top_k = 1
        top_p = 1.0
        temperature = 1.0

    if seed != "":
        try:
            torch.manual_seed(int(seed))
        except:
            return "❌ Seed non valido", "", "", "", "", None, None

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"][0]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_token,
            do_sample=not deterministico,
            temperature=temperature,
            top_k=0 if top_k == 0 else int(top_k),
            top_p=top_p,
            repetition_penalty=penalty,
            pad_token_id=tokenizer.eos_token_id
        )

    new_ids = outputs[0][len(input_ids):]
    testo_generato = tokenizer.decode(new_ids, skip_special_tokens=True)
    token_info = ' '.join(f"[{tid.item()}]" for tid in new_ids)
    n_token = len(new_ids)

    # Stato fine generazione
    last_id = outputs[0][-1].item()
    eos_id = tokenizer.eos_token_id
    end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if last_id == eos_id:
        chiusura = "✅ Terminato con <eos>"
    elif last_id == end_turn_id:
        chiusura = "⚠️ Terminato con <end_of_turn> (token 106)"
    else:
        chiusura = "⚠️ Nessun token di fine trovato"

    # Embedding PCA 2D
    gen_ids_tensor = new_ids.unsqueeze(0).to("cuda")
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(gen_ids_tensor)[0]
    path_2d = "pca_output.png"
    crea_grafico_pca(embeddings, path_2d)

    # Grafico 3D evolutivo
    image_path_3d = "embedding_3d.png"
    crea_grafico_3d_evolutivo(input_ids, image_path_3d)

    return testo_generato.strip(), token_info, f"{n_token} token", chiusura, f"{max_token} token max", path_2d, image_path_3d

# Interfaccia Gradio
with gr.Blocks(title="TKP – Generazione LLM con Visualizzazione PCA & Layer 3D") as interfaccia:
    gr.Markdown("## 🧪 TKP: Test LLM con PCA + Evoluzione 3D")

    prompt = gr.Textbox(lines=2, placeholder="Inserisci prompt", label="📝 Prompt")

    with gr.Row():
        temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
        top_k = gr.Slider(0, 100, value=50, step=1, label="Top-k (0 = disattivato)")
        top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p")
        penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.1, label="Repetition Penalty")

    with gr.Row():
        max_token_slider = gr.Slider(10, 512, value=100, step=10, label="📏 Token massimi generabili")
        seed = gr.Textbox(placeholder="es. 42", label="Seed (opzionale)")
        deterministico = gr.Checkbox(label="🔒 Modalità deterministica", value=False)

    risposta = gr.Textbox(label="🧠 Risposta generata")
    token_id = gr.Textbox(label="🧩 Token generati (ID)")
    conta = gr.Textbox(label="📏 Totale token generati", interactive=False)
    stato = gr.Textbox(label="🔚 Stato fine generazione", interactive=False)
    max_vis = gr.Textbox(label="📈 Limite token impostato", interactive=False)

    grafico = gr.Image(label="📊 PCA 2D degli embedding")
    grafico3d = gr.Image(label="🧬 Evoluzione 3D dei token nei layer")

    bottone = gr.Button("🔁 Genera")

    bottone.click(
        fn=genera,
        inputs=[prompt, temperature, top_k, top_p, penalty, seed, deterministico, max_token_slider],
        outputs=[risposta, token_id, conta, stato, max_vis, grafico, grafico3d]
    )

interfaccia.launch()
