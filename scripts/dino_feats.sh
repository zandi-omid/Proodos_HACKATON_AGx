#!/bin/bash
#SBATCH -J dino
#SBATCH -p public
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH --array=0-3
#SBATCH -o dino_%A_%a.log
module load mamba/latest
source activate /scratch/ilolos/envs/dino
cd /scratch/ilolos/HACKATHON_AGX/CDL
export HF_HUB_CACHE=/scratch/ilolos/hf_cache HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
/scratch/ilolos/envs/dino/bin/python dino_feats.py
