#!/bin/bash
#SBATCH --job-name=ecg_ablation
#SBATCH --partition=nodes
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/beegfs/labs/weinstocklab/projects/ydon268/Collaboration/ECG/slurm_out/ablation_%j.out
set -e
D=/beegfs/labs/weinstocklab/projects/ydon268/Collaboration/ECG
cd $D
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

TAG=${1:?tag required, e.g. stage2_r16_k16}
~/.pixi/bin/pixi run python scripts/causal_ablation.py \
    --classifier_ckpt $D/results/classifier/best.pt \
    --sae_ckpt        $D/results/sae/${TAG}.pt \
    --ptbxl_root      $D/data/ptbxl \
    --shortlist       $D/results/interp/${TAG}_top_latents.json \
    --out_dir         $D/results/ablation \
    --tag             $TAG \
    --batch_size      32
