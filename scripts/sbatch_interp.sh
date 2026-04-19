#!/bin/bash
#SBATCH --job-name=ecg_interp
#SBATCH --partition=nodes
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --array=0-35
#SBATCH --output=/beegfs/labs/weinstocklab/projects/ydon268/Collaboration/ECG/slurm_out/interp_%A_%a.out
set -e
D=/beegfs/labs/weinstocklab/projects/ydon268/Collaboration/ECG
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

~/.pixi/bin/pixi run python scripts/feature_interp.py \
    --sae_ckpt   $D/results/sae/${TAG}.pt \
    --acts       $D/results/activations/${HOOK}_activations.h5 \
    --beat_meta  $D/results/activations/beat_meta.parquet \
    --clinical_gt $D/results/clinical_gt_records.parquet \
    --out_dir    $D/results/interp \
    --tag        $TAG
