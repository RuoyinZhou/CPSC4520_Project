#!/bin/bash
#SBATCH --job-name=ecg_sae
#SBATCH --partition=nodes
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-35
#SBATCH --output=/beegfs/labs/weinstocklab/projects/ydon268/Collaboration/ECG/slurm_out/sae_%A_%a.out
set -e
D=/beegfs/labs/weinstocklab/projects/ydon268/Collaboration/ECG
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
~/.pixi/bin/pixi run python scripts/sae_topk.py \
    --acts       $D/results/activations/${HOOK}_activations.h5 \
    --hook       $HOOK \
    --expansion  $EXP --k $KVAL \
    --batch_size 4096 --lr 1e-4 --epochs 20 \
    --out_dir    $D/results/sae
