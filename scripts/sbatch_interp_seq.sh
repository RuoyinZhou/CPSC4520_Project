#!/bin/bash
#SBATCH --job-name=ecg_interp_seq
#SBATCH --partition=day
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/cpsc4520_project/logs/interp_seq_%j.out
set -e
module load Python/3.12.3-GCCcore-13.3.0 PyTorch/2.7.1-foss-2024a-CUDA-12.6.0 SciPy-bundle/2024.05-gfbf-2024a h5py/3.12.1-foss-2024a matplotlib/3.9.2-gfbf-2024a scikit-learn/1.5.2-gfbf-2024a
D=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/cpsc4520_project
cd $D
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

for HOOK in stage1 stage2 stage3; do
  for EXP in 4 8 16 32; do
    for KVAL in 8 16 32; do
      TAG="${HOOK}_r${EXP}_k${KVAL}"
      if [ ! -f $D/results/sae/${TAG}.pt ]; then
        echo "[skip] no checkpoint: ${TAG}"
        continue
      fi
      if [ -f $D/results/interp/${TAG}_interp_binary.parquet ]; then
        echo "[skip] already done: ${TAG}"
        continue
      fi
      echo "=== ${TAG} ==="
      python3 scripts/feature_interp.py \
          --sae_ckpt   $D/results/sae/${TAG}.pt \
          --acts       $D/results/activations/${HOOK}_activations.h5 \
          --beat_meta  $D/results/activations/beat_meta.parquet \
          --clinical_gt $D/results/clinical_gt_records.parquet \
          --out_dir    $D/results/interp \
          --tag        $TAG || echo "[fail] $TAG"
    done
  done
done
echo "[done] sequential interp"
