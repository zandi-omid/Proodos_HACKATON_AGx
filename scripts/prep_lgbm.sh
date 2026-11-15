#!/bin/bash
# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
#SBATCH -J prep
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 1:00:00
#SBATCH -o prep_%j.log
module load mamba/latest
source activate LST
cd /scratch/ilolos/HACKATHON_AGX/CDL
/home/ilolos/.conda/envs/LST/bin/python -W ignore prep_lgbm.py
