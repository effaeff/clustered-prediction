#!/bin/bash -l

#SBATCH --time=48:00:00
#SBATCH -N 1 -c 20
#SBATCH --partition=long

#SBATCH --mem=60G

#SBATCH --mail-user=felix.finkeldey@tu-dortmund.de
#SBATCH --mail-type=ALL
#SBATCH -o /work/smfefink/force2defl/out_bgmm_defl.txt
cd /work/smfefink/force2defl
export PYTHONPATH=$PYTHONPATH:/work/smfefink/pylib:/work/smfefink/pytorch-utils

module purge
module load python/3.9.10-wip
echo "sbatch: START SLURM_JOB_ID $SLURM_JOB_ID (SLURM_TASK_PID $SLURM_TASK_PID) on $SLURMD_NODENAME"
echo "sbatch: SLURM_JOB_NODELIST $SLURM_JOB_NODELIST"
echo "sbatch: SLURM_JOB_ACCOUNT $SLURM_JOB_ACCOUNT"
python3 scripts/learn_defl.py

