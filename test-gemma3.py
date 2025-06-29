from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./gemma3-4b-it"
torch.manual_seed(142)

print("🔄 Caricamento modello...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 📝 PROMPT
prompt = "inventa una breve storia di una formica di circa 50 parole"

# 🔠 TOKENIZZAZIONE
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"][0]

print("\n🧾 Prompt originale:")
print(prompt)

print("\n🔢 Prompt tokenizzati (ID ➔ testo):")
for tid in input_ids:
    print(f"{tid.item():>5} ➔ {tokenizer.decode(tid)}")

# 🧠 GENERAZIONE
print("\n🧠 Generazione creativa (con seed fisso)...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

# 🧾 RISPOSTA
generated_ids = outputs[0][len(input_ids):]          # solo i token nuovi
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print("\n🔢 Token generati (ID ➔ testo):")
for tid in generated_ids:
    print(f"{tid.item():>5} ➔ {tokenizer.decode(tid)}")

print(f"\n📏 Token generati: {len(generated_ids)}")

# 🌼 RISPOSTA DECODIFICATA
print("\n🌼 Risposta generata:\n")
print(generated_text)

# 🔚 CONTROLLO FINALE
eos_id        = tokenizer.eos_token_id
end_turn_id   = tokenizer.convert_tokens_to_ids("<end_of_turn>")  # di solito 106

last_id = outputs[0][-1].item()

if last_id == eos_id:
    print("\n✅ Generazione conclusa naturalmente con <eos>")
elif last_id == end_turn_id:
    print("\n⚠️ Generazione conclusa con <end_of_turn> (token 106), non con <eos>")
else:
    print("\n⚠️ Generazione troncata: nessun <eos> né <end_of_turn> trovato")
