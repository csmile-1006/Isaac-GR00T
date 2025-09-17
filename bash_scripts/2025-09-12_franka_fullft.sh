#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=franka_fullft                                                                                                                                 
#SBATCH --output=/home/changyeon/slurm-logs/franka_fullft/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/franka_fullft/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --exclude=node5,node7
#SBATCH --gres=gpu:a6000:2  # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 96시간 실행

TASK_NAME=$1
NUM_DEMOS=$2
STEPS=$3
NUM_GPUS=${4:-2}
BATCH_SIZE=${5:-16}
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE))

BASE_PATH=/home/changyeon
source /home/changyeon/miniconda3/bin/activate gr00t
cd /home/changyeon/workspace/Isaac-GR00T

script="
    WANDB_PROJECT=droid_panda_wrist_gripper_fullft \
    python scripts/gr00t_finetune.py \
        --dataset-path ~/data/franka_dataset/${TASK_NAME}_num${NUM_DEMOS}/ \
        --num-gpus ${NUM_GPUS} \
        --output-dir ${BASE_PATH}/franka_ckpts/${TASK_NAME}/FULLFT_bs${TOTAL_BATCH_SIZE}_step${STEPS}_demo${NUM_DEMOS} \
        --max-steps ${STEPS} \
        --data-config droid_panda_wrist_gripper \
        --batch-size ${BATCH_SIZE} \
        --save-steps 5000 \
        --run-name FULLFT_bs${TOTAL_BATCH_SIZE}_step${STEPS}_demo${NUM_DEMOS}
"

echo $script
eval $script