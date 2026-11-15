#!/bin/bash
#SBATCH -J build
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH -o build_%j.log
module load mamba/latest
source activate LST
cd /scratch/ilolos/HACKATHON_AGX/CDL
/home/ilolos/.conda/envs/LST/bin/python -W ignore build_table.py --watch
