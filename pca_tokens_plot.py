from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ⚙️ CONFIGURA QUI
model_path = "./gemma3-4b-it"
prompt = "La formica dormiva nel giardino."

# 🔄 Caricamento
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

# 🔠 Tokenizzazione
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]

# 🧠 Hidden states
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    last_layer = outputs.hidden_states[-1][0].cpu().numpy()  # (seq_len, hidden_size)

# 📉 PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(last_layer)

# 🧾 Token associati
tokens = [tokenizer.decode([tid.item()]) for tid in input_ids[0]]

# 📊 Plot
plt.figure(figsize=(10, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], color='blue')

for i, txt in enumerate(tokens):
    plt.annotate(txt, (reduced[i, 0], reduced[i, 1]), fontsize=12)

plt.title("Distribuzione semantica dei token (PCA 2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.tight_layout()
plt.show()
