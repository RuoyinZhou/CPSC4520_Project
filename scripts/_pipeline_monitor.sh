#!/bin/bash
# Watches SLURM + auto-submits downstream jobs when train finishes.
D=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/cpsc4520_project
LOG=$D/logs/pipeline.log
TS=$(date +%F_%T)

JOB_STATE=$(sacct -j 2197754 --format=State -P -n 2>/dev/null | head -1)
echo "[$TS] train 2197754 state=$JOB_STATE" >> $LOG

# If train finished successfully and no extract submitted yet, submit extract
if [[ "$JOB_STATE" == COMPLETED* ]] && [ ! -f $D/results/activations/stage1_activations.h5 ] && ! squeue -u cpsc4520_rz396 --name=ecg_extract -h | grep -q .; then
  EXT_JOB=$(sbatch --parsable $D/scripts/sbatch_extract_activations.sh 2>&1)
  echo "[$TS] submitted extract job $EXT_JOB" >> $LOG
fi

# If extract finished and no SAE submitted yet, submit SAE sweep
if [ -f $D/results/activations/stage3_activations.h5 ] && [ -f $D/results/activations/beat_meta.parquet ] && ! squeue -u cpsc4520_rz396 --name=ecg_sae -h | grep -q . && ! ls $D/results/sae/stage1_r4_k8.pt 2>/dev/null; then
  SAE_JOB=$(sbatch --parsable $D/scripts/sbatch_sae_sweep.sh 2>&1)
  echo "[$TS] submitted SAE sweep job $SAE_JOB" >> $LOG
fi

# If SAE sweep fully done and no interp running, submit interp
N_SAE=$(ls $D/results/sae/*.pt 2>/dev/null | wc -l)
if [ "$N_SAE" -eq 36 ] && ! squeue -u cpsc4520_rz396 --name=ecg_interp -h | grep -q . && ! ls $D/results/interp/stage1_r4_k8_interp_binary.parquet 2>/dev/null; then
  INT_JOB=$(sbatch --parsable $D/scripts/sbatch_interp.sh 2>&1)
  echo "[$TS] submitted interp job $INT_JOB" >> $LOG
fi
