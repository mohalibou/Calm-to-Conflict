# Calm-to-Conflict

Utterance-level conflict onset prediction in dyadic video interactions. This repository contains the Multimodal Transformer (MulT) implementation alongside code for audio feature extraction, text feature extraction, and baseline models (Early Fusion MLP, Feature-Level LSTM, Gemini model embedding).

---

## Repository Layout

```
Calm-To-Conflict/
├── MulT.ipynb                       # Multimodal Transformer
├── MulT_hpc_scripts/
│   ├── run_all_ablations.sh         # SLURM array job dispatcher (22 experiments)
│   ├── shared_utils.py              # Shared imports and model primitives
│   ├── sec2_dyadic.py               # 4-stream modality ablation
│   ├── sec3_visual.py               # Visual sub-feature ablation
│   ├── sec4_fusion.py               # Audio + FAU fusion (2 & 3-stream)
│   └── sec5_fusion.py               # Audio + FAU + body fusion (5-stream)
├── MulT_Plots/                      # Training curves & confusion matrices
├── MulT_results/                    # Per-experiment JSON results for MulT
├── Mult_results_summary.csv         # Summary of results for all 22 MulT experiments
└── MulT_master_results.json         # Combined per-experiment JSON results for MulT
```

---
## Video Feature Extraction

[TO-DO]

---

## Audio Feature Extraction

[TO-DO]

---

## Text Feature Extraction

[TO-DO]

---

## Baseline Models

### Early Fusion MLP

[TO-DO]

### Feature-Level LSTM

[TO-DO]

### Gemini Model Embedding

[TO-DO]

---

## MulT.ipynb

Readable version of the full MulT pipeline. Five sections run end-to-end.

| Section | Description |
|---------|-------------|
| **0 — Shared Config** | Imports, paths, seeds, model primitives (`PositionalEncoding`, `AttentionPool`, `CrossModalAttentionBlock`, cosine LR schedule) |
| **1 — Video Extraction** | Extracts 92-dim per-frame visual sequences (FAU + head + gaze + body) from `.npz` files → `video_self_sequences_v1.pt`, `video_partner_sequences_v1.pt`, `video_feature_dims.json` |
| **2 — Dyadic MulT Ablation** | 4-stream cross-modal Transformer (Text + Audio + Self-Video + Partner-Video); 9 ablation experiments |
| **3 — Visual Feature Ablation** | Decomposes 92-dim visual stream into FAU (24d), head (3d), gaze (2d), body (63d); 10 experiments |
| **4 — Optimized Audiovisual Fusion** | Audio (HuBERT) + FAU in a 3-stream dyadic model and 2-stream speaker-only baseline |
| **5 — 5-Stream Fusion** | Audio + Speaker FAU + Speaker Body + Partner FAU + Partner Body |

Each section follows: **Dataset → MixUp → Model → Training Loop → Ablation Runner** over 5 seeds, saving best-seed `.pt` weights.

---

## HPC Scripts

Production version of the notebook refactored for the **USC CARC HPC cluster** (Partition: `gpu`, GPU: A40). Runs all 22 ablation experiments in parallel via a SLURM array job (up to 4 concurrent).

**`shared_utils.py`** — imported by every section script; contains everything from Section 0: paths, seed utilities, data loaders, and all model primitives.

**`run_all_ablations.sh`** — SLURM array job mapping each index to one `(script, experiment)` pair:

```
Index  Script              Experiment
──────────────────────────────────────────────────
0–8    sec2_dyadic.py      Full_Dyadic, No_Partner, SelfVideo_PartnerVideo, …
9–18   sec3_visual.py      Full_Visual_Dyadic, FAU_Only, Head_Only, …
19–20  sec4_fusion.py      Audio_DyadicFAU, Audio_FAU_Only
21     sec5_fusion.py      Audio_DyadicFAUBody
```

```bash
# Submit all jobs
sbatch run_all_ablations.sh

# Or run a single experiment locally
python sec2_dyadic.py --experiment Full_Dyadic
python sec5_fusion.py --experiment Audio_DyadicFAUBody
```

---

## Architecture

### Core: `CrossModalAttentionBlock`

Standard cross-attention with residual connection + FFN. Query from one modality, keys/values from another.

### Section 2 — `CalmToConflict_DyadicMulT`

```
Text ──┬──→ T×A  ──→ T×VS  ──┐
Audio ─┼──→ A×T  ──→ A×VS  ──┤
       │                      ├──→ Concat → Classifier
SelfV ─┼──→ VS×T → VS×A   ──┤
       │    VS×VP ────────────┤
PartV ─┴──→ VP×VS ────────────┘
```
Gated attention pooling per stream → 4 × 128-dim → 512-dim fused representation.

### Section 5 — `Audio_DyadicFAUBody_MulT`

```
Audio   → A×FAU   → A×Body                    ──┐
SelfFAU → FAU×A   → FAU×Body → FAU×PartFAU   ──┤
SelfBody→ Bod×A   → Bod×FAU  → Bod×PartBody  ──┼──→ Concat → Classifier
PartFAU → FAU×SelfFAU                         ──┤
PartBody→ Bod×SelfBody                         ──┘
```
5 × 64-dim → 320-dim fused representation.

---

## Training

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (lr=1e-4, wd=1e-3) |
| LR Schedule | Cosine with 1-epoch warmup |
| Epochs | 25 (early stopping, patience=10) |
| Batch size | 64 |
| Augmentation | MixUp (α=0.2) |
| Label smoothing | 0.05 |
| Loss | Class-weighted soft cross-entropy |
| Evaluation | Macro F1 + Accuracy across 5 seeds |

---

## Embeddings

| File | Modality | Model | Dim |
|------|----------|-------|-----|
| `text_sequences_v1.pt` | Text (transcripts) | EmoRoBERTa | 768 |
| `audio_sequences_v2.pt` | Prosody | HuBERT-ER | 768 |
| `video_self_sequences_v1.pt` | Speaker visual | SEAMLESS | 92 |
| `video_partner_sequences_v1.pt` | Partner visual | SEAMLESS | 92 |

Visual stream (92-dim): FAU `[0:24]` · Head orientation `[24:27]` · Gaze `[27:29]` · Body pose `[29:92]`

---

## Outputs

```
out/
├── models/    # Best-seed .pt weights per experiment
├── plots/     # Loss curves + confusion matrices
└── results/   # Aggregated JSON (mean ± std across seeds)
```

Result JSON format:
```json
{
  "Full_Dyadic": {
    "acc_mean": 0.82, "acc_std": 0.01,
    "f1_mean": 0.81,  "f1_std": 0.01,
    "acc_per_seed": [...],
    "f1_per_seed": [...],
    "seeds": [42, 68, 92, 105, 208]
  }
}
```

---

## Environment

Cluster: **USC CARC** · Partition: `gpu` · GPU: `A40` · Python: `3.11` · CUDA: `12.1`

**Key Dependencies:**
`torch==2.5.1+cu121` | `transformers==5.2.0` | `librosa==0.11.0` | `seamless-interaction`

**Setup Instructions:**

```bash
# 1. Load Conda on the cluster
module load conda

# 2. Create the environment
conda env create -f environment.yml

# 3. Activate the environment
conda activate calm_conflict
```