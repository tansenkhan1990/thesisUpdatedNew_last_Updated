#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --time=80:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=4  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=3875M   # memory per CPU core
#SBATCH -J "500dimfb15k237-FieldEImplicit-six"   # job name
#SBATCH -o 500dimfb15k237-Implicit-six%j.out
#SBATCH --mail-user=my_mail   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --reservation=p_ml_nimi_219
#SBATCH -A p_ml_nimi
#SBATCH --array=1-24


source /home/mial610c/envs/env01/bin/activate

srun $(head -n $SLURM_ARRAY_TASK_ID commands.txt | tail -n 1)

exit 0