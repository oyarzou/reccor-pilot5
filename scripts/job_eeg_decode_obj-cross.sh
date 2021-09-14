#!/bin/bash

#SBATCH --mail-user=pablooyarzo@zedat.fu-berlin.de
#SBATCH --job-name=eeg-decod-obj-cross
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=100000
#SBATCH --time=14-00:00:00
#SBATCH --qos=standard


#down here goes the script
module add Anaconda3/2018.12

source activate imaging_env

declare -a combinations
index=0
for subject in 0 1
do
  for train in 'challenge' 'control'
  do
    for test in 'challenge' 'control'
    do
      combinations[$index]="$subject $train $test"
      index=$((index + 1))
    done
  done
done

parameters=(${combinations[${SLURM_ARRAY_TASK_ID}]})

subject=${parameters[0]}
train=${parameters[1]}
test=${parameters[2]}

python eeg_pipeline.py ${subject} 1 2 0 ${train} ${test}
