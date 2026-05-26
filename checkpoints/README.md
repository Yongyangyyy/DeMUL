# Pre-trained Checkpoints

Official DeMUL weights trained on TVR and DiDeMo validation splits.

| Dataset | Checkpoint | Config | Metrics |
|---------|------------|--------|---------|
| TVR | [demul_tvr/model.ckpt](demul_tvr/model.ckpt) | [opt.json](demul_tvr/opt.json) | [metrics.json](demul_tvr/metrics.json) |
| DiDeMo | [demul_didemo/model.ckpt](demul_didemo/model.ckpt) | [opt.json](demul_didemo/opt.json) | [metrics.json](demul_didemo/metrics.json) |

Each folder contains:
- `model.ckpt` — model weights (~65 MB)
- `opt.json` — training / inference options
- `metrics.json` — validation metrics (VCMR / SVMR / VR, NMS=0.7)

## Quick Use

```bash
# TVR
bash scripts/inference.sh checkpoints/demul_tvr val

# DiDeMo
bash scripts/inference.sh checkpoints/demul_didemo val
```

Or download via script:

```bash
bash scripts/download_weights.sh
```
