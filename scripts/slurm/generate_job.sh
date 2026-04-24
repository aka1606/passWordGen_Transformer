#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --job-name=pwdgen_generate
#SBATCH --output=output/results/slurm_gen_%j.log
#SBATCH --error=output/results/slurm_gen_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=akselalik2002@gmail.com

echo "=== Generation Job $SLURM_JOB_ID | $(date) ==="
echo "Noeud: $SLURMD_NODENAME"
nvidia-smi

cd $SLURM_SUBMIT_DIR
echo "Repertoire: $(pwd)"

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pwdgen

echo "=== Debut generation: $(date) ==="
python3 -u scripts/training/train_server.py --generate-only 2>&1
echo "=== Fin: $(date) ==="
