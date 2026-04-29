import math
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MASTER_CSV  = "../../utterance_level/out/master_dataset.csv"
OUTPUT_FILE = "out/text_sequences_v2.pt"

MODEL_NAME  = "j-hartmann/emotion-english-distilroberta-base"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(MASTER_CSV)

word_counts  = df["transcript"].astype(str).apply(lambda x: len(x.split()))
MAX_TEXT_LEN = math.ceil(word_counts.quantile(0.95) * 1.3)
print(f"MAX_TEXT_LEN: {MAX_TEXT_LEN} tokens")

print(f"Loading {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

text_dict = {}

with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        sample_id  = row["sample_id"]
        transcript = str(row["transcript"]).strip() or "[SILENCE]"

        inputs = tokenizer(
            transcript,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TEXT_LEN,
            padding="max_length",
        ).to(DEVICE)

        sequence = model(**inputs).last_hidden_state.squeeze(0).cpu()
        text_dict[sample_id] = sequence

torch.save(text_dict, OUTPUT_FILE)
print(f"Saved {len(text_dict)} sequences to {OUTPUT_FILE}")
print(f"Tensor shape per sample: ({MAX_TEXT_LEN}, 768)")
