#!/usr/bin/env bash
# Download pre-trained DeMUL checkpoints from GitHub.
# Usage: bash scripts/download_weights.sh [tvr|didemo|all]

set -euo pipefail

REPO="https://github.com/Yongyangyyy/DeMUL"
BRANCH="main"
TARGET=${1:-"all"}

download_one() {
  local name=$1
  local dir="checkpoints/${name}"
  mkdir -p "${dir}"
  echo "Downloading ${name} checkpoint..."
  curl -L "${REPO}/raw/${BRANCH}/${dir}/model.ckpt" -o "${dir}/model.ckpt"
  curl -L "${REPO}/raw/${BRANCH}/${dir}/opt.json" -o "${dir}/opt.json"
  curl -L "${REPO}/raw/${BRANCH}/${dir}/metrics.json" -o "${dir}/metrics.json"
  echo "Saved to ${dir}/"
}

case "${TARGET}" in
  tvr)    download_one demul_tvr ;;
  didemo) download_one demul_didemo ;;
  all)
    download_one demul_tvr
    download_one demul_didemo
    ;;
  *)
    echo "Usage: bash scripts/download_weights.sh [tvr|didemo|all]"
    exit 1
    ;;
esac

echo "Done."
