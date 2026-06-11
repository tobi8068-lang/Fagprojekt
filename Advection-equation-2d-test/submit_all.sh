#!/bin/sh
# Submit all 170 PINN jobs as individual LSF jobs (34 configs × 5 seeds).
# DTU HPC does not support job arrays via #BSUB -J name[0-N].
#
# Usage:
#   bash submit_all.sh           # submit all 170 jobs
#   bash submit_all.sh 0 9       # submit only jobs 0-9 (for testing)
#
# Check status: bstat
# Kill all your jobs: bkill 0

START=${1:-0}
END=${2:-169}

mkdir -p logs results

for i in $(seq $START $END); do
    bsub -J "pinn_$i" \
         -q gpuv100 \
         -n 4 \
         -R "span[hosts=1]" \
         -gpu "num=1:mode=exclusive_process" \
         -R "rusage[mem=4GB]" \
         -M 4GB \
         -W 08:00 \
         -o "logs/pinn_%J.out" \
         -e "logs/pinn_%J.err" \
         <<EOF
module load cuda/11.6
module load python3/3.11.13
source ~/venv_pinn/bin/activate
cd ~/Fagprojekt/Advection-equation-2d-test
python train.py --job_id $i --n_seeds 5 --out_dir results
EOF
done

echo "Submitted jobs $START to $END  ($(( END - START + 1 )) jobs)"
echo "Check status with: bstat"
