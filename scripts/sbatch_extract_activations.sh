#!/bin/bash
#SBATCH --job-name=ecg_extract
#SBATCH --partition=nodes
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/beegfs/labs/weinstocklab/projects/ydon268/Collaboration/ECG/slurm_out/extract_%j.out
set -e
D=/beegfs/labs/weinstocklab/projects/ydon268/Collaboration/ECG
cd $D
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
~/.pixi/bin/pixi run python scripts/extract_activations.py \
    --ptbxl_root $D/data/ptbxl \
    --ckpt       $D/results/classifier/best.pt \
    --out_dir    $D/results/activations \
    --batch 64 --workers ${SLURM_CPUS_PER_TASK:-16}
