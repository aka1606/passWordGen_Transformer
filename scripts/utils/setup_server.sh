#!/bin/bash
# Setup environnement conda sur le serveur Heracles (CUDA 12.1, P100)
# Usage: bash scripts/setup_server.sh

set -e

echo "=============================="
echo "Setup PasswordTransformer v6"
echo "=============================="

# 1. Creer environnement conda Python 3.10
echo ""
echo "[1/4] Creation environnement conda pwdgen..."
conda create -n pwdgen python=3.10 -y

# 2. Activer et installer PyTorch CUDA 12.1
echo ""
echo "[2/4] Installation PyTorch 2.x + CUDA 12.1..."
conda run -n pwdgen pip install torch --index-url https://download.pytorch.org/whl/cu121
conda run -n pwdgen pip install numpy

# 3. Verifier GPU
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

# 4. Creer dossiers output si inexistants
echo ""
echo "[4/4] Creation dossiers..."
mkdir -p output

echo ""
echo "=============================="
echo "Setup termine!"
echo ""
echo "Pour lancer l'entrainement:"
echo "  conda activate pwdgen"
echo "  screen -S training"
echo "  python scripts/train_server.py 2>&1 | tee output/training_log.txt"
echo ""
echo "Pour reprendre apres deconnexion SSH:"
echo "  screen -r training"
echo "=============================="
