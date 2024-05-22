#!/bin/bash
#SBATCH --chdir /home/fkhayat/project-m2-2024-4-pack/datasets/sbatch_results
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=4:0:0
#SBATCH --cpus-per-task=2
#SBATCH --mem 64G
#SBATCH --account cs-552
#SBATCH --reservation cs-552
#SBATCH --qos cs-552

echo STARTED at `date`

cd /home/fkhayat/project-m2-2024-4-pack/datasets
python data_cleanup.py math/Posts.xml math/math.jsonl

echo FINISHED at `date`
