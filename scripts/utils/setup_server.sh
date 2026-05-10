#!/bin/bash
# Setup environnement conda sur le serveur (CUDA 12.1, P100)
# Usage: bash scripts/utils/setup_server.sh

set -e

echo "=============================="
echo "Setup PasswordTransformer v7"
echo "=============================="

echo ""
echo "[1/4] Creation environnement conda pwdgen..."
conda create -n pwdgen python=3.10 -y

echo ""
echo "[2/4] Installation PyTorch 2.x + CUDA 12.1..."
conda run -n pwdgen pip install torch --index-url https://download.pytorch.org/whl/cu121
conda run -n pwdgen pip install numpy

echo ""
echo "[3/4] Verification GPU..."
conda run -n pwdgen python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA dispo: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
"

echo ""
echo "[4/4] Creation dossiers output..."
mkdir -p output/models output/results output/generated

echo ""
echo "=============================="
echo "Setup termine!"
echo ""
echo "Etapes suivantes:"
echo "  conda activate pwdgen"
echo "  python scripts/preprocess/download_rockyou.py"
echo "  python scripts/preprocess/prepare_dataset.py --freq-weight"
echo "  sbatch scripts/slurm/job_v7.sh"
echo "=============================="
