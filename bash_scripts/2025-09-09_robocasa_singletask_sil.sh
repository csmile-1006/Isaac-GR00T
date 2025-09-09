#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00tn15_rc_single_sil                                                                                                                                 
#SBATCH --output=/home/changyeon/slurm-logs/gr00tn15_rc_single_sil/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/gr00tn15_rc_single_sil/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:a6000:4   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=16     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 96시간 실행

TASK_NAME=$1
NUM_DEMOS=$2
STEPS=$3

base_path=/home/changyeon/
source ${base_path}/miniconda3/bin/activate gr00t
cd ${base_path}/workspace/Isaac-GR00T

python scripts/gr00t_finetune.py \
    --dataset-path "${base_path}/data/robocasa_dataset/${TASK_NAME}_num${NUM_DEMOS}/" \
                   "${base_path}/gr00tn15_robocasa/rollout_demos/${TASK_NAME}_num${NUM_DEMOS}/" \
    --num-gpus 4 \
    --output-dir "${base_path}/ckpts/SIL/step${STEPS}_${TASK_NAME}_num${NUM_DEMOS}" \
    --max-steps "${STEPS}" \
    --data-config single_panda_gripper \
    --batch-size 8 \
    --save-steps 5000
