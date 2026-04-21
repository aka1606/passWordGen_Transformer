#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=pwdgen_v6
#SBATCH --output=output/results/slurm_%j.log
#SBATCH --error=output/results/slurm_%j.err

echo "=== Job $SLURM_JOB_ID | $(date) ==="
echo "Noeud: $SLURMD_NODENAME"
nvidia-smi

cd $SLURM_SUBMIT_DIR
echo "Repertoire: $(pwd)"

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pwdgen

echo "=== Debut entrainement: $(date) ==="
python3 scripts/training/train_server.py 2>&1
echo "=== Fin: $(date) ==="
