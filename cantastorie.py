
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 🔧 CONFIGURAZIONE
MODEL_PATH = "./gemma3-4b-it"
SEED = 142
TEMPERATURE = 1
TOP_P = 0.5
TOP_K = 15
MAX_NEW_TOKENS = 512
REPETITION_PENALTY = 1.1

# 📝 PROMPT LUNGO CON A CAPO AUTOMATICI
PROMPT = """<start_of_turn>user
Scrivimi una breve storia di una formica.
la storia dovrà essere di circa 50 parole.<end_of_turn>
<start_of_turn>model
Nel cuore del giardino, viveva una piccola formica di nome Pip. Ogni giorno,
Pip si metteva in viaggio per raccogliere briciole e nutrire la sua colonia.
Era un lavoro duro, ma Pip lo faceva con gioia,
sapendo che il suo piccolo contributo era importante per il bene comune.
Un giorno, Pip scoprì una grande goccia di miele,
abbastanza da nutrire tutta la sua famiglia!<end_of_tunr>
<start_of_turn>user
Quale era il nome del protagonista della storia?
Che cosa era?<end_of_turn>
<start_of_turn>model
 """

# 🔄 SETUP MODELLO
torch.manual_seed(SEED)
print("🔄 Caricamento modello...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 🔠 TOKENIZZAZIONE
inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"][0]

print("\n🧾 Prompt originale:")
print(PROMPT)
print(f"\n📏 Lunghezza prompt: {len(input_ids)} token")

# 🧠 GENERAZIONE
print("\n🧠 Generazione in corso...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id
    )

# 🧾 ESTRAZIONE RISPOSTA
generated_ids = outputs[0][len(input_ids):]
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"\n📏 Token generati: {len(generated_ids)}")

# 🌼 RISULTATO
print("\n🌼 Storia generata:")
print("=" * 60)
print(generated_text)
print("=" * 60)

# 🔚 CONTROLLO FINALE
eos_id = tokenizer.eos_token_id
end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
last_id = outputs[0][-1].item()

if last_id == eos_id:
    print("\n✅ Generazione conclusa naturalmente con <eos>")
elif last_id == end_turn_id:
    print("\n⚠️ Generazione conclusa con <end_of_turn>")
else:
    print("\n⚠️ Generazione troncata: nessun token di fine trovato")
