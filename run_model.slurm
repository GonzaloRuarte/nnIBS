#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --nodes=6 
#SBATCH --chdir="/home/gruarte/repos/nnIBS"
#SBATCH --partition=cpu
    
arg1=$(SLURM_ARRAY_TASK_ID * 6 - 5) 

arg2=$(SLURM_ARRAY_TASK_ID * 6)    

    
python3 main.py -dataset "Hybrid Search ET Dataset VS" --rng ${arg1} ${arg2}

  