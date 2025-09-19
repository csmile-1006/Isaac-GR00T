#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=franka_mt_sil                                                                                                                                 
#SBATCH --output=/home/changyeon/slurm-logs/franka_mt_sil/%A_%a.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/franka_mt_sil/%A_%a.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --exclude=node5,node7
#SBATCH --gres=gpu:a6000:2  # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 96시간 실행

NUM_DEMOS=$1
STEPS=$2
NUM_GPUS=${3:-2}
BATCH_SIZE=${4:-16}
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE))

BASE_PATH=/home/changyeon
source /home/changyeon/miniconda3/bin/activate gr00t
cd /home/changyeon/workspace/Isaac-GR00T

TASK_NAMES=(
    "peach_center,11"
    "banana_center,6"
    "hichew_center,7"
)

dataset_path=""
for TASK in ${TASK_NAMES[@]}; do
    TASK_NAME=$(echo $TASK | cut -d ',' -f 1)
    NUM_ROLLOUT_DEMOS=$(echo $TASK | cut -d ',' -f 2)
    dataset_path="${dataset_path} ${BASE_PATH}/data/franka_dataset/franka_rl_data/rollout_demos/${TASK_NAME}_num${NUM_ROLLOUT_DEMOS}/"
    dataset_path="${dataset_path} ${BASE_PATH}/data/franka_dataset/franka_rl_data/${TASK_NAME}_num${NUM_DEMOS}/"
done

script="
    WANDB_PROJECT=droid_panda_wrist_gripper_mt \
    python scripts/gr00t_finetune.py \
        --dataset-path ${dataset_path} \
        --num-gpus ${NUM_GPUS} \
        --output-dir ${BASE_PATH}/franka_ckpts/multiple/SIL_bs${TOTAL_BATCH_SIZE}_step${STEPS}_demo${NUM_DEMOS} \
        --max-steps ${STEPS} \
        --data-config droid_panda_wrist_gripper \
        --batch-size ${BATCH_SIZE} \
        --save-steps 10000 \
        --run-name SIL_bs${TOTAL_BATCH_SIZE}_step${STEPS}_demo${NUM_DEMOS} \
        --video_backend torchvision_av
"

echo $script
eval $script