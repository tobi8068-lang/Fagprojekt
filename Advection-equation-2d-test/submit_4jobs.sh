#!/bin/sh
# Submit 4 LSF jobs that together cover all 170 PINN configs×seeds.
# Each job: 1 GPU, 4 cores, runs its slice of job IDs sequentially.
# You receive an email when each job starts and when it finishes.
#
# Usage:
#   bash submit_4jobs.sh
#
# Splits:  part 0: jobs   0– 42  (43 jobs)
#          part 1: jobs  43– 85  (43 jobs)
#          part 2: jobs  86–127  (42 jobs)
#          part 3: jobs 128–169  (42 jobs)

EMAIL="s245200@dtu.dk"

mkdir -p logs results

# Array: start end  for each part
STARTS=(  0  43  86 128)
ENDS=(   42  85 127 169)

for part in 0 1 2 3; do
    S=${STARTS[$part]}
    E=${ENDS[$part]}

    bsub \
        -J "pinn_p${part}" \
        -q gpuv100 \
        -n 4 \
        -R "span[hosts=1]" \
        -gpu "num=1:mode=exclusive_process" \
        -R "rusage[mem=4GB]" \
        -M 4GB \
        -W 06:00 \
        -B \
        -N \
        -u "$EMAIL" \
        -o "logs/part${part}_%J.out" \
        -e "logs/part${part}_%J.err" \
        <<EOF
module load cuda/11.6
module load python3/3.11.13
source ~/venv_pinn/bin/activate
cd ~/Fagprojekt/Advection-equation-2d-test
mkdir -p logs results

echo "=== Part ${part}: jobs ${S}–${E} started at \$(date) ==="

for i in \$(seq $S $E); do
    echo "--- job \$i start \$(date) ---"
    python train.py --job_id \$i --n_seeds 5 --out_dir results \\
        > logs/run_\${i}.out 2>&1
    echo "--- job \$i done  \$(date) ---"
done

echo "=== Part ${part} finished at \$(date) ==="
EOF

    echo "Submitted part ${part}  (jobs ${S}–${E})"
done

echo ""
echo "4 jobs queued. Check status: bstat"
echo "Cancel all:    bkill 0"
