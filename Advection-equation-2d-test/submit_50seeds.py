"""
Submit 50-seed runs for two high-variance MoE configs:
  config 23: moe_cont_rff0_fd0_sa0_ar1_lb1
  config 26: moe_cont_rff0_fd0_sa1_ar0_lb1

100 jobs total, split across 4 LSF jobs (1 GPU each).

Usage (on HPC login node):
    python submit_50seeds.py
"""
import subprocess

EMAIL     = "s245200@dtu.dk"
N_PARTS   = 4
OUT_DIR   = "results_50seeds"

# Config indices and 50 seeds each (non-overlapping with main sweep seeds)
CONFIGS_50 = [23, 26]
SEEDS_50   = list(range(10000, 10050))  # 50 seeds: 10000–10049

# Build flat list of (config_idx, seed) pairs
jobs = [(c, s) for c in CONFIGS_50 for s in SEEDS_50]

print(f"Configs : {CONFIGS_50}")
print(f"Seeds   : {SEEDS_50[0]}–{SEEDS_50[-1]}")
print(f"Total   : {len(jobs)} jobs")
print()

# Round-robin split into N_PARTS batches
batches = [jobs[k::N_PARTS] for k in range(N_PARTS)]

for part, batch in enumerate(batches):
    if not batch:
        continue

    cmds = "\n".join(
        f"python train.py --config_idx {c} --seed {s} --out_dir {OUT_DIR}"
        for c, s in batch
    )

    script = f"""\
#!/bin/sh
#BSUB -J pinn_50s_p{part}
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 12:00
#BSUB -B
#BSUB -N
#BSUB -u {EMAIL}
#BSUB -o logs/50s_p{part}_%J.out
#BSUB -e logs/50s_p{part}_%J.err

module load cuda/11.6
module load python3/3.11.13
source ~/venv_pinn/bin/activate
cd ~/Fagprojekt/Advection-equation-2d-test
mkdir -p logs {OUT_DIR}

echo "=== 50-seed part {part}: {len(batch)} runs  started at $(date) ==="

{cmds}

echo "=== 50-seed part {part} finished at $(date) ==="
"""

    result = subprocess.run(["bsub"], input=script, text=True, capture_output=True)
    out = result.stdout.strip()
    err = result.stderr.strip()
    c0, s0 = batch[0];  c1, s1 = batch[-1]
    print(f"Part {part}  ({len(batch)} runs, cfg{c0}/seed{s0}–cfg{c1}/seed{s1}):  {out}")
    if err:
        print(f"  stderr: {err}")

print()
print("Done. Check status: bstat")
print("Cancel all:    bkill 0")
