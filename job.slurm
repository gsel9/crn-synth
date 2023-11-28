#!/bin/bash
#SBATCH --job-name=nrwe

#SBATCH --account=p1849_tsd
##SBATCH --reservation=submit

#SBATCH --partition=accel
#SBATCH --gres=gpu:1

#SBATCH --mem-per-cpu=200G
#SBATCH --time=08:00:00

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

module purge

#module load Anaconda3/5.3.8

set -o errexit

srun python synthesis_scripts/nrwe/nrwe_synthesis_main.py
#srun python evaluation_scripts/nrwe/score_reports.py
#srun python synthesis_scripts/nrwe/nrwe_report.py
