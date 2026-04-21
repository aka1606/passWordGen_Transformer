#!/bin/bash
# A lancer UNE SEULE FOIS en interactif (pas via sbatch)
# Usage: bash scripts/utils/setup_env.sh

set -e
cd "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"  # project root

echo "=== [1/4] Conda env ==="
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda create -n pwdgen python=3.10 -y
conda activate pwdgen

echo "=== [2/4] PyTorch CUDA 12.1 ==="
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy

echo "=== [3/4] Verification GPU ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo "=== [4/4] Preparation des donnees RockYou ==="
mkdir -p output/models output/results output/generated
python3 scripts/utils/prepare_data.py

echo ""
echo "Setup termine. Lance maintenant: sbatch scripts/utils/job.sh"
