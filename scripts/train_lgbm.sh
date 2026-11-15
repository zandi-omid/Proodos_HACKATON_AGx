#!/bin/bash
#SBATCH -J train
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o train_%j.log
module load mamba/latest
source activate LST
cd /scratch/ilolos/HACKATHON_AGX/CDL
/home/ilolos/.conda/envs/LST/bin/python -W ignore train_lgbm.py
