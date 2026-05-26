# DeMUL

Official implementation of **Video Corpus Moment Retrieval via Decoupled Multimodal Modeling and Unified Localization**.

DeMUL is a clean, minimal re-implementation that keeps only the core DeMUL model and removes experimental / alternative modules from the research codebase.

---

## Highlights

- **Decoupled multimodal encoding**: separate visual and subtitle encoders with cross-modal attention from the query
- **Query-driven fusion**: NetVLAD-based modality weighting with missing-modality fallback
- **Unified localization**: shared moment localization head for VCMR / SVMR / VR tasks
- **Supported datasets**: [TVR](https://github.com/jayleicn/TVRetrieval) and [DiDeMo](https://github.com/LisaAnne/LocalizingMoments)

---

## Project Structure

```
DeMUL/
├── model/
│   ├── transformer.py   # LayerNorm, MaskedMHA/MHCA, LocalMaskedMHCA, TransformerBlock
│   ├── layers.py        # LinearLayer, NetVLAD, ConvSE
│   ├── encoder.py       # BidVideoQueryEncoder, QueryWeightEncoder
│   ├── head.py          # MomentLocalizationHead (BiGRU + ConvSE)
│   └── demul.py         # DeMUL main model
├── config/
│   ├── config.py
│   ├── model_config.json
│   ├── tvr_data_config.json
│   └── didemo_data_config.json
├── data_loader/
│   └── dataset.py
├── optim/
│   └── adamw.py
├── utils/
├── standalone_eval/
│   └── eval.py
├── train.py
├── inference.py
├── checkpoints/         # pre-trained TVR weights
└── scripts/
    ├── train_tvr.sh
    ├── train_didemo.sh
    ├── inference.sh
    └── download_weights.sh
```

---

## Model Architecture

```
Input: (query, visual clips, subtitle clips)
          │
          ▼
  BidVideoQueryEncoder
  ┌──────────────────────────────────────────────────────────┐
  │  Linear projection  (visual 4352→384, sub 768→384,       │
  │                       query 768→384)                     │
  │  + position & token-type embeddings                      │
  │                                                          │
  │  queryEncoder   : TransformerBlock (full self-attn)      │
  │  visualEncoder  : TransformerBlock (win-5 self-attn       │
  │                     + cross-modal attn from query)       │
  │  textEncoder    : TransformerBlock (same as visual)      │
  └──────────────────────────────────────────────────────────┘
          │
          ▼
  QueryWeightEncoder  (NetVLAD → sigmoid → per-modality weights)
          │
          ▼
  Contextual QDF Refinement  (2 × TransformerBlock, win-5 + cross-modal)
          │
          ▼
  MoE Weighted Fusion  +  missing-modality fallback
          │
          ▼
  MomentLocalizationHead  (BiGRU × 2 + ConvSE × 2)
          │
          ▼
  start / end score distributions  →  Cross-Entropy loss
```

---

## Installation

```bash
git clone https://github.com/Yongyangyyy/DeMUL.git
cd DeMUL

# Option 1: pip
pip install -r requirements.txt

# Option 2: conda (recommended for CUDA)
conda create -n demul python=3.10 -y
conda activate demul
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Key dependencies: `torch>=2.0`, `tensorboard`, `lmdb`, `msgpack`, `msgpack-numpy`, `easydict`, `tqdm`.

---

## Data Preparation

This repo expects pre-extracted features in the same format as the [DeMUL/DeMA feature release](https://github.com/Yongyangyyy/DeMUL). After downloading the features, update `root_path` in the dataset config:

```bash
# TVR
vim config/tvr_data_config.json   # set "root_path" to your feature directory

# DiDeMo
vim config/didemo_data_config.json
```

Each dataset config uses paths relative to `root_path` for annotations, LMDB features, and VR rank lists.

---

## Model Zoo

We release the best pre-trained **TVR** checkpoint. Metrics are reported on the **validation** split with NMS threshold 0.7.

| Dataset | VCMR R@1<br/>(IoU=0.7) | VCMR R@5<br/>(IoU=0.7) | SVMR R@1<br/>(IoU=0.7) | VR R@1 | Download |
|:-------:|:----------------------:|:----------------------:|:----------------------:|:------:|:--------:|
| TVR | 11.92 | 25.27 | 26.84 | 29.01 | [model.ckpt](checkpoints/demul_tvr/model.ckpt) |

> Full metrics and training config are bundled in [`checkpoints/demul_tvr/`](checkpoints/demul_tvr/).

### Download

**Option 1 — script**

```bash
bash scripts/download_weights.sh
```

**Option 2 — manual**

```bash
wget https://github.com/Yongyangyyy/DeMUL/raw/main/checkpoints/demul_tvr/model.ckpt \
     -P checkpoints/demul_tvr/
wget https://github.com/Yongyangyyy/DeMUL/raw/main/checkpoints/demul_tvr/opt.json \
     -P checkpoints/demul_tvr/
wget https://github.com/Yongyangyyy/DeMUL/raw/main/checkpoints/demul_tvr/metrics.json \
     -P checkpoints/demul_tvr/
```

### Inference with Pre-trained Weights

```bash
bash scripts/inference.sh checkpoints/demul_tvr val
```

---

## Training

### TVR

```bash
bash scripts/train_tvr.sh demul_tvr_run1
```

Or directly:

```bash
python train.py \
    --exp_id my_run \
    --dset_name tvr \
    --dataset_config config/tvr_data_config.json \
    --model_config   config/model_config.json
```

### DiDeMo

```bash
bash scripts/train_didemo.sh demul_didemo_run1
```

To fine-tune from a TVR checkpoint:

```bash
bash scripts/train_didemo.sh demul_didemo_run1 2018 0 results/tvr-my_run-YYYY_MM_DD_HH_MM_SS
```

---

## Inference

```bash
bash scripts/inference.sh results/tvr-my_run-YYYY_MM_DD_HH_MM_SS val
```

Checkpoints and logs are saved under `results/<exp_name>-<timestamp>/`.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Citation

If you find this code useful, please cite:

```bibtex
@article{demul2026,
  title={Video Corpus Moment Retrieval via Decoupled Multimodal Modeling and Unified Localization},
  author={Yang, Yongyang},
  year={2026}
}
```

---

## Acknowledgements

This implementation builds upon ideas and data pipelines from prior VCMR works including [MINUTE](https://arxiv.org/abs/2301.13606) and [PREM](https://arxiv.org/abs/2402.13576).
