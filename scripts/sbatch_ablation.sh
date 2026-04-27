#!/bin/bash
#SBATCH --job-name=ecg_ablation
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/cpsc4520_project/logs/ablation_%j.out
set -e
module load Python/3.12.3-GCCcore-13.3.0 PyTorch/2.7.1-foss-2024a-CUDA-12.6.0 SciPy-bundle/2024.05-gfbf-2024a h5py/3.12.1-foss-2024a matplotlib/3.9.2-gfbf-2024a scikit-learn/1.5.2-gfbf-2024a
D=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/cpsc4520_project
cd $D
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

TAG=${1:?tag required, e.g. stage2_r16_k16}
python3 scripts/causal_ablation.py \
    --classifier_ckpt $D/results/classifier/best.pt \
    --sae_ckpt        $D/results/sae/${TAG}.pt \
    --ptbxl_root      $D/data/ptbxl \
    --shortlist       $D/results/interp/${TAG}_top_latents.json \
    --out_dir         $D/results/ablation \
    --tag             $TAG \
    --batch_size      32
