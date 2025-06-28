from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import softmax

# === CONFIG ===
model_path = "./gemma3-4b-it"

print("🔄 Caricamento modello...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.eval()

# === INPUT UTENTE ===
prompt = input("📝 Prompt: ").strip() or "scrivi una frase"
seed = int(input("🌱 Seed (default 42): ") or 42)
temperature = float(input("🌡️ Temperatura (default 0.8): ") or 0.8)
top_k = int(input("🔝 Top-K (default 50): ") or 50)
top_p = float(input("📊 Top-P (default 0.9): ") or 0.9)
max_tokens = int(input("📏 Token da generare (default 20): ") or 20)

torch.manual_seed(seed)

# === TOKEN INPUT ===
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]

# === GENERAZIONE ===
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.1,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id
    )

# === ESTRAZIONE ===
gen_ids = output.sequences[0][len(input_ids[0]):]
logits_seq = output.scores

print("\n🔍 Token generati e probabilità top-10 per ciascuno:\n")

for i, (logit_step, token_id) in enumerate(zip(logits_seq, gen_ids)):
    probs = softmax(logit_step[0], dim=0)
    top10 = torch.topk(probs, 10)
    top_ids = top10.indices
    top_probs = top10.values

    actual_id = token_id.item()
    actual_text = tokenizer.decode([actual_id])

    print(f"🧩 Token {i+1}: '{actual_text}'  (ID={actual_id})")
    for rank, (tid, prob) in enumerate(zip(top_ids, top_probs), 1):
        ttxt = tokenizer.decode([tid])
        mark = "✅" if tid.item() == actual_id else "  "
        print(f"   {mark} {rank:2d}. '{ttxt}' (ID={tid.item():<6}) → {prob:.4f}")
    print()

# === TESTO COMPLETO ===
text_out = tokenizer.decode(gen_ids, skip_special_tokens=True)
print(f"\n📝 Risultato:\n{text_out}\n")
