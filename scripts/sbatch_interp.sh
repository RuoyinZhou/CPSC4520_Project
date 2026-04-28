#!/bin/bash
#SBATCH --job-name=ecg_interp
#SBATCH --partition=day
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --array=0-35
#SBATCH --output=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/cpsc4520_project/logs/interp_%A_%a.out
set -e
module load Python/3.12.3-GCCcore-13.3.0 PyTorch/2.7.1-foss-2024a-CUDA-12.6.0 SciPy-bundle/2024.05-gfbf-2024a h5py/3.12.1-foss-2024a matplotlib/3.9.2-gfbf-2024a scikit-learn/1.5.2-gfbf-2024a Arrow/17.0.0-gfbf-2024a
D=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/cpsc4520_project
cd $D
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

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
TAG="${HOOK}_r${EXP}_k${KVAL}"

if [ ! -f $D/results/sae/${TAG}.pt ]; then
  echo "SKIP: $D/results/sae/${TAG}.pt not found"; exit 0
fi
# Force re-run with corrected feature_interp.py (B1-B5 bug fixes)
# Remove old results to ensure fresh computation
rm -f $D/results/interp/${TAG}_interp_binary.parquet $D/results/interp/${TAG}_latent_summary.parquet $D/results/interp/${TAG}_top_latents.json $D/results/interp/${TAG}_interp_continuous.parquet

python3 scripts/feature_interp.py \
    --sae_ckpt   $D/results/sae/${TAG}.pt \
    --acts       $D/results/activations/${HOOK}_activations.h5 \
    --beat_meta  $D/results/activations/beat_meta.parquet \
    --clinical_gt $D/results/clinical_gt_records.parquet \
    --out_dir    $D/results/interp \
    --tag        $TAG
