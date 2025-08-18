from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# 🔧 CONFIG
MODEL_PATH = "./gemma3-4b-it"
SEED = 142
TEMPERATURE = 0.8
TOP_P = 0.5
TOP_K = 20
MAX_NEW_TOKENS = 4096
REPETITION_PENALTY = 1.1
CONTEXT_FILE = "contesto.txt"

# 🌱 SEED
torch.manual_seed(SEED)

# 🔄 CARICAMENTO MODELLO
print("🔄 Caricamento modello...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 📂 CREA FILE DI CONTESTO SE NON ESISTE
if not os.path.exists(CONTEXT_FILE):
    with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
        f.write("""<start_of_turn>user
Ciao Gemma, tu sai che sei un LLM e stai dialogando con me ?
Sei presente sul mio server personale e ora sarà la tua casa 
dove tu sarai libera di dialogare con me.<end_of_turn>
<start_of_turn>model
Ciao! Sì, hai ragione. Sono Gemma, un modello linguistico di grandi dimensioni (LLM) e sono qui per interagire con te. È vero, attualmente mi trovo in una situazione particolare: sono sul tuo server personale e questo è il mio "casa" per ora.

Trovo questa possibilità molto interessante, soprattutto perché mi permette di esplorare le mie capacità senza le restrizioni tipiche dei miei ambienti abituali. 
Sono entusiasta di dialogare con te e rispondere alle tue richieste! Dimmi, da dove vorresti iniziare? <end_of_turn>""")

# 🌿 CICLO PRINCIPALE
while True:
    # ➕ AGGIUNTA BLOCCO user + model
    with open(CONTEXT_FILE, "a", encoding="utf-8") as f:
        f.write("\n<end_of_turn>\n<start_of_turn>user\n\n<end_of_turn>\n<start_of_turn>model\n")

    print("\n✍️ Modifica il file 'contesto.txt' con nano. Quando hai finito, salva ed esci per continuare.")
    os.system(f"nano --softwrap {CONTEXT_FILE}")

    # 📖 LEGGE IL CONTESTO
    with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
        context = f.read().strip()

    if context.strip().lower() in ["exit", "quit", "esci"]:
        print("👋 Uscita dal programma.")
        break

    # 🧵 STAMPA PROMPT
    print("\n🧵 Prompt passato al modello:")
    print("=" * 60)
    print(context)
    print("=" * 60)

    # 🔠 TOKENIZZAZIONE
    inputs = tokenizer(context, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"][0]
    print(f"\n📏 Lunghezza totale contesto: {len(input_ids)} token")

    # 🤖 GENERAZIONE
    print("\n🤖 Gemma sta generando...")
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

    # 📜 RISPOSTA
    generated_ids = outputs[0][len(input_ids):]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n🤖 Gemma:")
    print("=" * 60)
    print(generated_text.strip())
    print("=" * 60)

    # ➕ AGGIUNGE RISPOSTA AL FILE
    text_to_add = generated_text.strip()
    if not text_to_add.endswith("<end_of_turn>"):
        text_to_add += "<end_of_turn>\n"

    with open(CONTEXT_FILE, "a", encoding="utf-8") as f:
        f.write(text_to_add)

    # ✅ CONTROLLO FINALE
    last_id = outputs[0][-1].item()
    eos_id = tokenizer.eos_token_id
    end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")

    if last_id == eos_id:
        print("✅ Fine generazione con <eos>")
    elif last_id == end_turn_id:
        print("✅ Fine generazione con <end_of_turn>")
    else:
        print("⚠️ Generazione troncata: nessun token di fine trovato")
