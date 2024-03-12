#!/bin/bash

#SBATCH --job-name=run_pipeline
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="abbo@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="cpu-long"
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --mem=40G

module load Python/3.11.3-GCCcore-12.3.0

source $HOME/data1/mrp/venv/bin/activate

python data_analysis.py