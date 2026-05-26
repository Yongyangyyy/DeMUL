# Pre-trained Checkpoints

Official DeMUL TVR checkpoint (validation split, seed=2012).

| Checkpoint | Config | Metrics |
|------------|--------|---------|
| [demul_tvr/model.ckpt](demul_tvr/model.ckpt) | [opt.json](demul_tvr/opt.json) | [metrics.json](demul_tvr/metrics.json) |

Each folder contains:
- `model.ckpt` — model weights (~65 MB)
- `opt.json` — training / inference options
- `metrics.json` — validation metrics (VCMR / SVMR / VR, NMS=0.7)

## Quick Use

```bash
bash scripts/inference.sh checkpoints/demul_tvr val
```

Or download via script:

```bash
bash scripts/download_weights.sh
```
