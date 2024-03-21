#!/bin/bash

#SBATCH --job-name=run_pipeline_load_pickle
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="abbo@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="cpu-medium"
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --mem=50G

module load Python/3.11.3-GCCcore-12.3.0

source $HOME/data1/mrp/venv/bin/activate

rm -r plots/
mkdir plots

python data_analysis.py -l

cat run_pipeline_$SLURM_JOB_ID.out > plots/run_output.txt