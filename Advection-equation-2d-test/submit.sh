#!/bin/sh
# Single-job template used by submit_all.sh.
# Not meant to be submitted directly — use submit_all.sh instead.
#
# Check queue status: bstat

### Job name (overridden per-job by submit_all.sh)
#BSUB -J pinn_job

### GPU queue
#BSUB -q gpuv100

### Cores + GPU
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

### Memory: 4 GB per core
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB

### Walltime — GPU queues max 24 h
#BSUB -W 08:00

### Output files
#BSUB -o logs/pinn_%J.out
#BSUB -e logs/pinn_%J.err

### Email on completion
#BSUB -N
#BSUB -u s245200@dtu.dk

# ---------------------------------------------------------------------------
module load cuda/11.6
module load python3/3.11.13
source ~/venv_pinn/bin/activate

mkdir -p logs results

python train.py \
    --job_id  "$LSB_JOBINDEX" \
    --n_seeds 5 \
    --out_dir results
