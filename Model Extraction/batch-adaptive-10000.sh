#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-48:00      # time (DD-HH:MM)
#SBATCH --output=adaptive_10000.out  # %N for node name, %j for jobID
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zaheerkhan23599@gmail.com
​
module load python/3.6
module load cuda cudnn 
source ~/jupyter_py3/bin/activate
export PYTHONPATH="${PYTHONPATH}:/home/mohan235/projects/def-guzdial/mohan235/CMPUT622_project/knockoffnets"
python ~/projects/def-guzdial/mohan235/CMPUT622_project/knockoffnets/batch_model_adaptive_10000.py