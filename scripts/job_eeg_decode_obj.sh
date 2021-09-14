#!/bin/bash

#SBATCH --mail-user=pablooyarzo@zedat.fu-berlin.de
#SBATCH --job-name=eeg-decod-obj
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=100000
#SBATCH --time=14-00:00:00
#SBATCH --qos=standard


#down here goes the script
module add Anaconda3/2018.12

source activate imaging_env

python eeg_pipeline.py ${SLURM_ARRAY_TASK_ID} 1 1 0 'control' 'control'
