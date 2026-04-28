#!/bin/bash
#SBATCH --job-name=ecg_train
#SBATCH --partition=education_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/ECG/logs/train_%j.out
set -e
module load Python/3.12.3-GCCcore-13.3.0 PyTorch/2.7.1-foss-2024a-CUDA-12.6.0 SciPy-bundle/2024.05-gfbf-2024a h5py/3.12.1-foss-2024a matplotlib/3.9.2-gfbf-2024a scikit-learn/1.5.2-gfbf-2024a Arrow/17.0.0-gfbf-2024a
python3 -c "import torch; print(f'[CUDA] available={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}', flush=True)"
D=/nfs/roberts/project/cpsc4520/cpsc4520_rz396/ECG
cd $D
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
python3 scripts/train_classifier.py \
    --ptbxl_root $D/data/ptbxl \
    --out_dir    $D/results/classifier_v2 \
    --epochs 200 --bs 64 --lr 1e-3 --patience 50 --early_stop 50 --workers 0
