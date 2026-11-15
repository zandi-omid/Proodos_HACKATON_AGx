#!/bin/bash
#SBATCH -J prep
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 1:00:00
#SBATCH -o prep_%j.log
module load mamba/latest
source activate LST
cd /scratch/ilolos/HACKATHON_AGX/CDL
/home/ilolos/.conda/envs/LST/bin/python -W ignore prep_lgbm.py
