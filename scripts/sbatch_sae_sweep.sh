#!/bin/bash
#SBATCH --job-name=ecg_sae
#SBATCH --partition=day
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-35
#SBATCH --output=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/cpsc4520_project/logs/sae_%A_%a.out
set -e
module load Python/3.12.3-GCCcore-13.3.0 PyTorch/2.7.1-foss-2024a-CUDA-12.6.0 SciPy-bundle/2024.05-gfbf-2024a h5py/3.12.1-foss-2024a matplotlib/3.9.2-gfbf-2024a scikit-learn/1.5.2-gfbf-2024a
D=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/cpsc4520_project
cd $D
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# 3 hooks x 4 expansions x 3 k = 36 configs
HOOKS=(stage1 stage2 stage3)
EXPS=(4 8 16 32)
KS=(8 16 32)
IDX=${SLURM_ARRAY_TASK_ID}
H=$(( IDX / (4 * 3) ))
REM=$(( IDX % (4 * 3) ))
E=$(( REM / 3 ))
K=$(( REM % 3 ))
HOOK=${HOOKS[$H]}
EXP=${EXPS[$E]}
KVAL=${KS[$K]}
echo "config: hook=$HOOK expansion=$EXP k=$KVAL"
python3 scripts/sae_topk.py \
    --acts       $D/results/activations/${HOOK}_activations.h5 \
    --hook       $HOOK \
    --expansion  $EXP --k $KVAL \
    --batch_size 4096 --lr 1e-4 --epochs 20 \
    --out_dir    $D/results/sae
