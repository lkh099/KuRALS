#!/usr/bin/env bash
# Bootstrap a clean virtualenv for KuRALS, matching the environment this
# codebase is actually developed and tested against: Python 3.10, torch
# 2.3.1+cu121 / torchvision 0.18.1+cu121. Not conda -- this project runs from
# a plain venv.
#
# Usage: bash scripts/setup_env.sh [venv_path]
#   venv_path defaults to .venv in the repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${1:-$REPO_ROOT/.venv}"

python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

pip install --upgrade pip

# torch/torchvision come from the PyTorch CUDA wheel index, not PyPI -- if
# your GPU/driver needs a different CUDA build, adjust the index URL (see
# https://pytorch.org/get-started/previous-versions/ for other versions).
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r "$REPO_ROOT/requirements.txt"
pip install -e "$REPO_ROOT"

echo
echo "Environment ready at $VENV_PATH (activate with: source $VENV_PATH/bin/activate)"
echo
echo "Optional: the AdaPKC-Xi baseline variant (kuralsnet_adapkcxi) needs a"
echo "separately compiled 'correlation' CUDA extension, unrelated to the"
echo "current NPU/tracker work -- see README.md's install section if needed."
