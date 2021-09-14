#!/bin/bash

#SBATCH --mail-user=pablooyarzo@zedat.fu-berlin.de
#SBATCH --job-name=eeg-decod-img
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=12000
#SBATCH --time=4-00:00:00
#SBATCH --qos=standard


#down here goes the script
module add Anaconda3/2018.12

source activate imaging_env

declare -a combinations
index=0
for subject in 0 1
do
  for iteration in 1 2 3 4 5
  do
    combinations[$index]="$subject $iteration"
    index=$((index + 1))
  done
done

parameters=(${combinations[${SLURM_ARRAY_TASK_ID}]})

subject=${parameters[0]}
iteration=${parameters[1]}

python eeg_pipeline.py ${subject} 1 0 ${iteration} 'control' 'control'
