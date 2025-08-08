#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH -J remedy-anonymization
#SBATCH -p p_48G
#SBATCH -w gpu-a3090-01
#SBATCH -G 1

export GRADIO_TEMP_DIR="tmp/"
uv run app.py