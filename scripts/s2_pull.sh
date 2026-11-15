#!/bin/bash
#SBATCH -J s2_pull
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 2:00:00
#SBATCH --array=0-72
#SBATCH -o s2_pull_%A_%a.log
module load mamba/latest
source activate LST
cd /scratch/ilolos/HACKATHON_AGX/CDL
/home/ilolos/.conda/envs/LST/bin/python s2_pull.py
