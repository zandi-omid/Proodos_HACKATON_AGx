#!/bin/bash
#SBATCH -J ks_pull
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 3:00:00
#SBATCH --array=0-17
#SBATCH -o ks_pull_%A_%a.log
module load mamba/latest
source activate LST
cd /scratch/ilolos/HACKATHON_AGX/CDL
/home/ilolos/.conda/envs/LST/bin/python ks_pull.py 1
