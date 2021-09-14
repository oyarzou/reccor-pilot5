#!/bin/bash

#SBATCH --mail-user=pablooyarzo@zedat.fu-berlin.de
#SBATCH --job-name=meg-decod-obj-cross
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=150000
#SBATCH --time=14-00:00:00
#SBATCH --qos=standard


#down here goes the script
module add Anaconda3/2018.12

source activate imaging_env

declare -a combinations
index=0
for subject in 0 1
do
  for noise_norm in 0 1
  if [ "$noise"]

  if %noise_norm%==0
  do
    for decode in 0
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

python meg_pipeline.py ${subject} 1 4 0 ${train} ${test}
