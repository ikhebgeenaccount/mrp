#!/bin/bash

#SBATCH --job-name=download_maps
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="abbo@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="cpu-medium"
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --mem=10M

wget -r -np -R "index.html*" http://cuillin.roe.ac.uk/~bengib/MassMaps4JHD/ -nH --cut-dirs=2