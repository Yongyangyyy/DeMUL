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
└── scripts/
    ├── train_tvr.sh
    ├── train_didemo.sh
    └── inference.sh
```

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

## Acknowledgements

