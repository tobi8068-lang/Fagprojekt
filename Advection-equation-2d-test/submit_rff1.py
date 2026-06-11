"""
Submit re-runs for all rff1 configs (fixed B-matrix seed).
Splits 60 jobs (12 configs × 5 seeds) across 4 LSF jobs, 1 GPU each.
Email sent on start and finish of each LSF job.

Usage (on HPC login node):
    python submit_rff1.py
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from configs import CONFIGS, SEEDS

EMAIL    = "s245200@dtu.dk"
N_SEEDS  = len(SEEDS)
N_PARTS  = 4

# Collect job IDs for all rff1 configs
rff1_jobs = [
    i * N_SEEDS + j
    for i, cfg in enumerate(CONFIGS)
    if cfg.get("use_rff")
    for j in range(N_SEEDS)
]

print(f"rff1 configs : {len(rff1_jobs) // N_SEEDS}")
print(f"Total jobs   : {len(rff1_jobs)}")
print(f"Job IDs      : {rff1_jobs}")
print()

os.makedirs("logs",    exist_ok=True)
os.makedirs("results", exist_ok=True)

# Round-robin split into N_PARTS batches
batches = [rff1_jobs[k::N_PARTS] for k in range(N_PARTS)]

for part, batch in enumerate(batches):
    if not batch:
        continue

    ids = " ".join(str(j) for j in batch)

    script = f"""\
#!/bin/sh
#BSUB -J pinn_rff1_p{part}
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 04:00
#BSUB -B
#BSUB -N
#BSUB -u {EMAIL}
#BSUB -o logs/rff1_p{part}_%J.out
#BSUB -e logs/rff1_p{part}_%J.err

module load cuda/11.6
module load python3/3.11.13
source ~/venv_pinn/bin/activate
cd ~/Fagprojekt/Advection-equation-2d-test
mkdir -p logs results

echo "=== rff1 part {part}: {len(batch)} jobs  started at $(date) ==="

for i in {ids}; do
    echo "--- job $i start $(date) ---"
    python train.py --job_id $i --n_seeds 5 --out_dir results > logs/run_$i.out 2>&1
    echo "--- job $i done  $(date) ---"
done

echo "=== rff1 part {part} finished at $(date) ==="
"""

    result = subprocess.run(["bsub"], input=script, text=True,
                            capture_output=True)
    out = result.stdout.strip()
    err = result.stderr.strip()
    print(f"Part {part}  ({len(batch)} jobs, IDs {batch[0]}–{batch[-1]}):  {out}")
    if err:
        print(f"  stderr: {err}")

print()
print("Done. Check status: bstat")
print("Cancel all:    bkill 0")
