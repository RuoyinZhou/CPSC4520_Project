#!/bin/bash
#SBATCH --job-name=ecg_extract
#SBATCH --partition=day
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/cpsc4520_project/logs/extract_%j.out
set -e
module load Python/3.12.3-GCCcore-13.3.0 PyTorch/2.7.1-foss-2024a-CUDA-12.6.0 SciPy-bundle/2024.05-gfbf-2024a h5py/3.12.1-foss-2024a matplotlib/3.9.2-gfbf-2024a scikit-learn/1.5.2-gfbf-2024a
D=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/cpsc4520_project
cd $D
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
python3 scripts/extract_activations.py \
    --ptbxl_root $D/data/ptbxl \
    --ckpt       $D/results/classifier/best.pt \
    --out_dir    $D/results/activations \
    --batch 64 --workers ${SLURM_CPUS_PER_TASK:-16}
