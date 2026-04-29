import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

AUDIO_PATH = "../../MULT_V2/embeddings/audio_sequences_v2.pt"
TEXT_PATH  = "../emoroberta_text/out/text_sequences_v2.pt"
VIDEO_PATH = "../../MULT_V2/embeddings/video_self_sequences_v1.pt"
LABEL_PATH = "../../MULT_V2/out/master_dataset_dyadic.csv"
OUTPUT_DIR = "out"

SEED       = 100
T_TARGET   = 45
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
print(f"Device: {DEVICE}")


def resize_sequence(seq, target_len):
    T, D = seq.shape
    if T == target_len:
        return seq
    if T > target_len:
        idx = torch.linspace(0, T - 1, steps=target_len).long()
        return seq[idx]
    pad = torch.zeros(target_len - T, D, dtype=seq.dtype)
    return torch.cat([seq, pad], dim=0)


class EarlyFusionDataset(Dataset):
    def __init__(self, sample_ids, audio_dict, text_dict, video_dict, label_map):
        self.sample_ids = sample_ids
        self.audio_dict = audio_dict
        self.text_dict  = text_dict
        self.video_dict = video_dict
        self.label_map  = label_map

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]

        audio = resize_sequence(self.audio_dict[sid].float(), T_TARGET)
        text  = resize_sequence(self.text_dict[sid].float(),  T_TARGET)
        video = resize_sequence(self.video_dict[sid].float(), T_TARGET)

        fused  = torch.cat([audio, text, video], dim=-1)
        pooled = fused.mean(dim=0)

        label = self.label_map[sid]
        return pooled, torch.tensor(label, dtype=torch.long)


class EarlyFusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(y.tolist())
    return accuracy_score(labels, preds), f1_score(labels, preds, average="macro")


print("Loading embeddings...")
text_dict   = torch.load(TEXT_PATH,  map_location="cpu")
video_dict  = torch.load(VIDEO_PATH, map_location="cpu", weights_only=False)
audio_obj   = torch.load(AUDIO_PATH, map_location="cpu")
audio_dict  = audio_obj["audio_sequences"]

labels_df   = pd.read_csv(LABEL_PATH)
label_map   = dict(zip(labels_df["sample_id"].astype(str), labels_df["label"].astype(int)))
file_id_map = dict(zip(labels_df["sample_id"].astype(str), labels_df["file_id"].astype(str)))

common_ids = sorted(set(audio_dict) & set(text_dict) & set(video_dict) & set(label_map))
print(f"Common samples: {len(common_ids)}")

groups = np.array([file_id_map[sid] for sid in common_ids])
ids_np = np.array(common_ids)

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_val_idx, test_idx = next(gss1.split(ids_np, groups=groups))

train_val_ids = ids_np[train_val_idx]
test_ids      = ids_np[test_idx].tolist()

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)
train_idx, val_idx = next(gss2.split(train_val_ids, groups=np.array([file_id_map[s] for s in train_val_ids])))

train_ids = train_val_ids[train_idx].tolist()
val_ids   = train_val_ids[val_idx].tolist()

print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

train_ds = EarlyFusionDataset(train_ids, audio_dict, text_dict, video_dict, label_map)
val_ds   = EarlyFusionDataset(val_ids,   audio_dict, text_dict, video_dict, label_map)
test_ds  = EarlyFusionDataset(test_ids,  audio_dict, text_dict, video_dict, label_map)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

INPUT_DIM = 768 + 768 + 92
model     = EarlyFusionMLP(input_dim=INPUT_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_val_f1 = -1.0
best_path   = os.path.join(OUTPUT_DIR, "best_early_fusion_mlp.pt")

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x.to(DEVICE)), y.to(DEVICE))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_acc, val_f1 = evaluate(model, val_loader)
    print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_path)
        print(f"  Saved new best model.")

model.load_state_dict(torch.load(best_path, map_location=DEVICE))
test_acc, test_f1 = evaluate(model, test_loader)

print(f"\nFINAL TEST RESULTS | Acc: {test_acc:.4f} | Macro F1: {test_f1:.4f}")

with open(os.path.join(OUTPUT_DIR, "early_fusion_mlp_results.txt"), "w") as f:
    f.write(f"Test Accuracy: {test_acc:.6f}\n")
    f.write(f"Test Macro F1: {test_f1:.6f}\n")

print("Results saved to", OUTPUT_DIR)
