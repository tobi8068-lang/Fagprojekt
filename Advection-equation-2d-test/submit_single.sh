#!/bin/sh
#BSUB -J pinn_sweep
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=4:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -W 24:00
#BSUB -o logs/pinn_sweep_%J.out
#BSUB -e logs/pinn_sweep_%J.err
#BSUB -N
#BSUB -u s245200@dtu.dk

module load cuda/11.6
module load python3/3.11.13
source ~/venv_pinn/bin/activate
cd ~/Fagprojekt/Advection-equation-2d-test
mkdir -p logs results

N_GPU=4
pids=()

for i in $(seq 0 209); do
    gpu_id=$(( i % N_GPU ))

    echo "=== Launching job $i on GPU $gpu_id  $(date) ==="
    CUDA_VISIBLE_DEVICES=$gpu_id \
        python train.py --job_id $i --n_seeds 5 --out_dir results \
        > logs/run_${i}.out 2>&1 &

    pids+=($!)

    # Once N_GPU jobs are running, wait for all before launching the next batch
    if (( (i + 1) % N_GPU == 0 )); then
        for pid in "${pids[@]}"; do
            wait $pid
        done
        pids=()
        echo "=== Batch ending at job $i done  $(date) ==="
    fi
done

# Wait for any remaining jobs in the last partial batch
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All done at $(date)"
