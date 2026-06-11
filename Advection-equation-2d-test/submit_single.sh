#!/bin/sh
# Single LSF job that runs all 170 configs×seeds in parallel across N_GPU GPUs.
# Each GPU runs one job at a time; when it finishes the next job starts on it.
#
# Usage:
#   bsub < submit_single.sh
#
# Adjust N_GPU and -W walltime to match your allocation.
# Rough estimate: 170 jobs / N_GPU * ~10 min each.
#   N_GPU=4  → ~7 h    N_GPU=8  → ~3.5 h

#BSUB -J pinn_sweep
#BSUB -q gpuv100
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=4:mode=exclusive_process"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 12:00
#BSUB -o logs/pinn_sweep_%J.out
#BSUB -e logs/pinn_sweep_%J.err
#BSUB -N
#BSUB -u s245200@dtu.dk

module load cuda/11.6
module load python3/3.11.13
source ~/venv_pinn/bin/activate
cd ~/Fagprojekt/Advection-equation-2d-test
mkdir -p logs results

N_GPU=4      # must match num= in #BSUB -gpu above
N_JOBS=170   # 34 configs × 5 seeds

# Run jobs in parallel batches of N_GPU.
# Each job is pinned to one GPU via CUDA_VISIBLE_DEVICES.
pids=()

for i in $(seq 0 $(( N_JOBS - 1 ))); do
    gpu_id=$(( i % N_GPU ))

    echo "=== Launching job $i on GPU $gpu_id  $(date) ==="
    CUDA_VISIBLE_DEVICES=$gpu_id \
        python train.py --job_id $i --n_seeds 5 --out_dir results \
        > logs/run_${i}.out 2>&1 &

    pids+=($!)

    # Once a full batch of N_GPU jobs is running, wait for all before next batch
    if (( (i + 1) % N_GPU == 0 )); then
        for pid in "${pids[@]}"; do wait $pid; done
        pids=()
        echo "=== Batch ending at job $i done  $(date) ==="
    fi
done

# Wait for any remaining jobs in the last partial batch
for pid in "${pids[@]}"; do wait $pid; done

echo "=== All $N_JOBS jobs done at $(date) ==="
