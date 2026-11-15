#!/bin/bash
# Spray or Wait? - Hack Arizona AgX 2026, challenge 1 (pesticide drift / temperature inversions).
# Written by team Ioannis Lolos & Omid Zandi during the hackathon (15 Nov 2026), with help from an
# AI coding assistant. Timeline and who did what: README, section "Built during the hackathon".
# Orchard-only 2025 CDL at 10 m, read from the Source Cooperative Zarr copy (no zip download).
# Usage:  cd /scratch/ilolos/HACKATHON_AGX/CDL && sbatch cdl_orchards_zarr.sh
#SBATCH -J cdl_orchards_zarr
#SBATCH -p public
#SBATCH -q public
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-12:00:00
#SBATCH -o cdl_orchards_zarr.out

module load mamba/latest
source activate LST
export PYTHONNOUSERSITE=1
export PROJ_DATA=$HOME/.conda/envs/LST/share/proj

cd /scratch/ilolos/HACKATHON_AGX/CDL || exit 1
curl -s -m 30 -o /dev/null https://data.source.coop || { echo "this node cannot reach data.source.coop"; exit 1; }
$HOME/.conda/envs/LST/bin/python cdl_orchards_zarr.py 8
