#!/usr/bin/env bash
# Download the pre-trained TVR checkpoint from GitHub.
# Usage: bash scripts/download_weights.sh

set -euo pipefail

REPO="https://github.com/Yongyangyyy/DeMUL"
BRANCH="main"
DIR="checkpoints/demul_tvr"

mkdir -p "${DIR}"
echo "Downloading TVR checkpoint..."
curl -L "${REPO}/raw/${BRANCH}/${DIR}/model.ckpt" -o "${DIR}/model.ckpt"
curl -L "${REPO}/raw/${BRANCH}/${DIR}/opt.json" -o "${DIR}/opt.json"
curl -L "${REPO}/raw/${BRANCH}/${DIR}/metrics.json" -o "${DIR}/metrics.json"
echo "Saved to ${DIR}/"
echo "Done."
