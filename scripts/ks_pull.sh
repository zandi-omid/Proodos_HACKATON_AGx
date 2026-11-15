#!/bin/bash
# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
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
