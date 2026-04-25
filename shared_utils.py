import os
import json
import random
import copy
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns

PATHS = {
    "csv":     "out/master_dataset_dyadic.csv",
    "models":  "out/models",
    "plots":   "out/plots",
    "results": "out/results",
}

EMBEDDINGS = {
    "text":            "embeddings/text_sequences_v1.pt",
    "audio":           "embeddings/audio_sequences_v2.pt",
    "video_self":      "embeddings/video_self_sequences_v1.pt",
    "video_partner":   "embeddings/video_partner_sequences_v1.pt",
    "video_feat_dims": "embeddings/video_feature_dims.json",
}

for path in PATHS.values():
    if not path.endswith(".csv"):
        os.makedirs(path, exist_ok=True)

RANDOM_SEEDS = [42, 68, 92, 105, 208]

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def group_split(df: pd.DataFrame, test_size: float = 0.20, seed: int = 42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df["file_id"].values
    train_idx, test_idx = next(gss.split(df, groups=groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

def load_embedding(key: str):
    raw = torch.load(EMBEDDINGS[key], map_location="cpu", weights_only=False)
    return raw.get("audio_sequences", raw) if isinstance(raw, dict) else raw

def aggregate_seeds(acc_list: list, f1_list: list) -> dict:
    acc_arr, f1_arr = np.array(acc_list), np.array(f1_list)
    return {
        "acc_mean": float(acc_arr.mean()), "acc_std": float(acc_arr.std()),
        "f1_mean": float(f1_arr.mean()), "f1_std": float(f1_arr.std()),
        "acc_per_seed": [float(x) for x in acc_arr],
        "f1_per_seed": [float(x) for x in f1_arr],
        "seeds": list(RANDOM_SEEDS),
    }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x.transpose(0, 1))

class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)
    def forward(self, x):
        w = torch.softmax(self.score(x), dim=1)
        return (x * w).sum(dim=1)

class CrossModalAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * 2, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, query_seq, key_value_seq):
        attn_out, _ = self.cross_attn(query=query_seq, key=key_value_seq, value=key_value_seq)
        x = self.norm1(query_seq + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def save_training_plots(exp_name, train_losses, val_losses, val_accuracies, best_epoch, final_preds, final_targets):
    plot_dir = PATHS["plots"]
    epochs_run = len(train_losses)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs_run + 1), train_losses, label="Train Loss", color="steelblue", linewidth=2)
    plt.plot(range(1, epochs_run + 1), val_losses, label="Val Loss", color="tomato", linewidth=2, linestyle="--")
    plt.axvline(x=best_epoch, color="green", linestyle=":", label=f"Best {best_epoch}")
    plt.title(f"Loss: {exp_name}"); plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs_run + 1), [a * 100 for a in val_accuracies], label="Val Accuracy", color="mediumseagreen", linewidth=2)
    plt.axvline(x=best_epoch, color="red", linestyle=":", label=f"Best {best_epoch}")
    plt.title(f"Accuracy: {exp_name}"); plt.xlabel("Epochs"); plt.ylabel("Accuracy (%)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{exp_name}_curves.png"), dpi=200)
    plt.close()
    
    cm = confusion_matrix(final_targets, final_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Calm", "Predicted Conflict"], yticklabels=["Actual Calm", "Actual Conflict"], annot_kws={"size": 14})
    plt.title(f"Confusion Matrix: {exp_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{exp_name}_cm.png"), dpi=200)
    plt.close()