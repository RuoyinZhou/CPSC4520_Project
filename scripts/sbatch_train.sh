#!/bin/bash
#SBATCH --job-name=ecg_train
#SBATCH --partition=nodes
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/beegfs/labs/weinstocklab/projects/ydon268/Collaboration/ECG/slurm_out/train_%j.out
set -e
D=/beegfs/labs/weinstocklab/projects/ydon268/Collaboration/ECG
cd $D
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
~/.pixi/bin/pixi run python scripts/train_classifier.py \
    --ptbxl_root $D/data/ptbxl \
    --out_dir    $D/results/classifier_v2 \
    --epochs 200 --bs 64 --lr 1e-3 --patience 50 --early_stop 50 --workers 0
