#!/bin/bash
#SBATCH --job-name=ecg_sae
#SBATCH --partition=education_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/ECG/logs/sae_%j.out
set -e
module load Python/3.12.3-GCCcore-13.3.0 PyTorch/2.7.1-foss-2024a-CUDA-12.6.0 SciPy-bundle/2024.05-gfbf-2024a h5py/3.12.1-foss-2024a matplotlib/3.9.2-gfbf-2024a scikit-learn/1.5.2-gfbf-2024a Arrow/17.0.0-gfbf-2024a
python3 -c "import torch; print(f'[CUDA] available={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}', flush=True)"
D=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/ECG
cd $D
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# 3 hooks x 4 expansions x 3 k = 36 configs (sequential due to QOS MaxSubmitPU=1)
HOOKS=(stage1 stage2 stage3)
EXPS=(4 8 16 32)
KS=(8 16 32)
for HOOK in "${HOOKS[@]}"; do
  for EXP in "${EXPS[@]}"; do
    for KVAL in "${KS[@]}"; do
      TAG="${HOOK}_r${EXP}_k${KVAL}"
      if [ -f $D/results/sae/${TAG}.pt ]; then
        echo "[skip] already done: ${TAG}"
        continue
      fi
      echo "=== config: hook=$HOOK expansion=$EXP k=$KVAL ==="
      python3 scripts/sae_topk.py \
          --acts       $D/results/activations/${HOOK}_activations.h5 \
          --hook       $HOOK \
          --expansion  $EXP --k $KVAL \
          --batch_size 4096 --lr 1e-4 --epochs 20 \
          --out_dir    $D/results/sae
    done
  done
done
echo "[done] all 36 SAE configs"
