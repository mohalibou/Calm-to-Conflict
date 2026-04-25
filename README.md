# Calm-to-Conflict — Dyadic Multimodal Transformer (MulT)

Utterance-level conflict onset prediction in dyadic video interactions using a cross-modal Transformer that fuses text, audio, and visual streams from both speakers.

---

## Repository Layout

```
FINAL/
├── MulT.ipynb                  # Full pipeline — read this first
├── hpc_scripts/                # SLURM-ready scripts executed on USC HPC
│   ├── run_all_ablations.sh    # SLURM array job dispatcher (22 experiments)
│   ├── shared_utils.py         # Shared imports, primitives, and helpers
│   ├── sec2_dyadic.py          # Section 2 — 4-stream modality ablation
│   ├── sec3_visual.py          # Section 3 — visual sub-feature ablation
│   ├── sec4_fusion.py          # Section 4 — audio + FAU fusion (2 & 3-stream)
│   └── sec5_fusion.py          # Section 5 — audio + FAU + body fusion (5-stream)
├── Plots/                      # Training curves & confusion matrices 
└── results/                    # Per-experiment JSON result files 
```

---

## MulT.ipynb — The Full Pipeline

The notebook is the canonical, readable version of the entire project. All five sections run end-to-end in a single file.

| Section | What it does |
|---------|-------------|
| **0 — Shared Config** | Imports, output paths, random seeds, model primitives (`PositionalEncoding`, `AttentionPool`, `CrossModalAttentionBlock`, cosine LR schedule) |
| **1 — Video Extraction** | Extracts 92-dim per-frame visual sequences (FAU + head + gaze + body) from raw `.npz` files; produces `video_self_sequences_v1.pt`, `video_partner_sequences_v1.pt`, `video_feature_dims.json` |
| **2 — Dyadic MulT Ablation** | 4-stream cross-modal Transformer (Text + Audio + Self-Video + Partner-Video); 9 modality-ablation experiments to isolate each stream's contribution |
| **3 — Visual Feature Ablation** | Decomposes the 92-dim visual stream into FAU (24d), head (3d), gaze (2d), body (63d); 10 experiments to identify which behaviors drive predictions |
| **4 — Optimized Audiovisual Fusion** | Combines Audio (HuBERT) + FAU in a 3-stream dyadic model and a 2-stream speaker-only baseline |
| **5 — 5-Stream Fusion** | Extends Section 4 with body pose: Audio + Speaker FAU + Speaker Body + Partner FAU + Partner Body |

Every section follows the same structure: **Dataset → MixUp → Model → Training Loop → Ablation Runner**. Each runner iterates over 5 random seeds and saves the single best-seed `.pt` weights.

---

## HPC Scripts — `hpc_scripts/`

The scripts in `hpc_scripts/` are the production version of the notebook, refactored for parallel execution on the **USC CARC HPC cluster** (Partition: `gpu`, GPU: A40).

### Why separate scripts?

The notebook is ideal for reading and development. For running 22 ablation experiments with 5 seeds each, a SLURM array job is far more practical — all 22 jobs dispatch in parallel (up to 4 concurrent) on the cluster.

### `shared_utils.py`

Imported by every section script. Contains everything from Section 0 of the notebook:
- `PATHS` and `EMBEDDINGS` dictionaries
- `set_seed`, `group_split`, `load_embedding`, `aggregate_seeds`
- `PositionalEncoding`, `AttentionPool`, `CrossModalAttentionBlock`
- `get_cosine_schedule_with_warmup`
- `save_training_plots`

### `run_all_ablations.sh`

SLURM array job covering all 22 experiments across sections 2–5. Each array index maps to one `(script, experiment_name)` pair.

```
Index  Script              Experiment
─────────────────────────────────────────────────
0-8    sec2_dyadic.py      Full_Dyadic, No_Partner, SelfVideo_PartnerVideo, …
9-18   sec3_visual.py      Full_Visual_Dyadic, FAU_Only, Head_Only, …
19-20  sec4_fusion.py      Audio_DyadicFAU, Audio_FAU_Only
21     sec5_fusion.py      Audio_DyadicFAUBody
```

To submit all jobs:
```bash
sbatch run_all_ablations.sh
```

Each script accepts an `--experiment` argument and can also be run locally:
```bash
python sec2_dyadic.py --experiment Full_Dyadic
python sec5_fusion.py --experiment Audio_DyadicFAUBody
```

---

## Models & Architecture

### Core Building Block — `CrossModalAttentionBlock`

Standard cross-attention with residual connection + FFN. Query comes from one modality stream, keys/values from another.

### Section 2 — `CalmToConflict_DyadicMulT`

```
Text ──┬──→ T×A ──→ T×VS ──┐
Audio ─┼──→ A×T ──→ A×VS ──┤
       │                    ├──→ Concat → Classifier
SelfV ─┼──→ VS×T → VS×A ──┤
       │    VS×VP ──────────┤
PartV ─┴──→ VP×VS ──────────┘
```
Gated attention pooling per stream. 4 × 128-dim → 512-dim fused representation.

### Section 5 — `Audio_DyadicFAUBody_MulT`

```
Audio ──→ A×FAU → A×Body ──────────────────┐
SelfFAU → FAU×A → FAU×Body → FAU×PartFAU ─┤
SelfBody→ Bod×A → Bod×FAU → Bod×PartBody ─┤──→ Concat → Classifier
PartFAU → FAU×SelfFAU ─────────────────────┤
PartBody→ Bod×SelfBody ─────────────────────┘
```
5 × 64-dim → 320-dim fused representation.

---

## Training Details

| Hyperparameter | Value |
|---------------|-------|
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
| `video_self_sequences_v1.pt` | Speaker visual | OpenFace / MediaPipe | 92 |
| `video_partner_sequences_v1.pt` | Partner visual | OpenFace / MediaPipe | 92 |

Visual stream breakdown (92-dim total): FAU (0:24) · Head orientation (24:27) · Gaze (27:29) · Body pose (29:92).

---

## Outputs

After running, outputs are written to:

```
out/
├── models/    # Best-seed .pt weights per experiment
├── plots/     # Loss curves + confusion matrices (.png)
└── results/   # Aggregated JSON (mean ± std across seeds)
```

Result JSON format:
```json
{
  "Full_Dyadic": {
    "acc_mean": 0.82, "acc_std": 0.01,
    "f1_mean":  0.81, "f1_std":  0.01,
    "acc_per_seed": [...],
    "f1_per_seed":  [...],
    "seeds": [42, 68, 92, 105, 208]
  }
}
```

---

## HPC Environment

Cluster: **USC CARC** · Partition: `gpu` · GPU: `A40` · Conda env: `calm_conflict`

```bash
module load conda
source activate calm_conflict
```