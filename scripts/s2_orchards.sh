#!/bin/bash
#SBATCH -J s2_orch
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 2:00:00
#SBATCH --array=0-39
#SBATCH -o s2_orch_%A_%a.log
module load mamba/latest
source activate LST
cd /scratch/ilolos/HACKATHON_AGX/CDL
/home/ilolos/.conda/envs/LST/bin/python s2_orchards.py
