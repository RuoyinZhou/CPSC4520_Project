#!/bin/bash
#SBATCH --job-name=ecg_fig2
#SBATCH --partition=day
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/cpsc4520_project/logs/fig2_%j.out
set -e
module load Python/3.12.3-GCCcore-13.3.0 \
  PyTorch/2.7.1-foss-2024a-CUDA-12.6.0 \
  SciPy-bundle/2024.05-gfbf-2024a \
  Arrow/17.0.0-gfbf-2024a \
  h5py/3.12.1-foss-2024a \
  matplotlib/3.9.2-gfbf-2024a \
  scikit-learn/1.5.2-gfbf-2024a

D=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/cpsc4520_project
cd $D

python3 scripts/make_fig2_top_beats.py \
  --sae_ckpt   $D/results/sae/stage3_r32_k32.pt \
  --acts        $D/results/activations/stage3_activations.h5 \
  --beat_meta   $D/results/activations/beat_meta.parquet \
  --latent_summary $D/results/interp/stage3_r32_k32_latent_summary.parquet \
  --interp_binary  $D/results/interp/stage3_r32_k32_interp_binary.parquet \
  --ptbxl_root  $D/data/ptbxl \
  --fig_dir     $D/figures
