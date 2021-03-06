#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --time=40:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=4  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=3875M   # memory per CPU core
#SBATCH -J "complex_quad_first_run"   # job name
#SBATCH -o complex_command%j.out
#SBATCH --mail-user=tansenkhan1083@gmail.com   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_ml_nimi
#SBATCH --array=1-24


source /home/mdkh039d/envs/env001/bin/activate

srun $(head -n $SLURM_ARRAY_TASK_ID commands.txt | tail -n 1)

exit 0