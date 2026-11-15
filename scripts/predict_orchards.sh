#!/bin/bash
#SBATCH -J predict
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0:30:00
#SBATCH -o predict_%j.log
module load mamba/latest
source activate LST
cd /scratch/ilolos/HACKATHON_AGX/CDL
/home/ilolos/.conda/envs/LST/bin/python -W ignore predict_orchards.py
