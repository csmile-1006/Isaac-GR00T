#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=franka_mt_allft                                                                                                                                 
#SBATCH --output=/home/changyeon/slurm-logs/franka_mt_allft/%A_%a.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/franka_mt_allft/%A_%a.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --exclude=node5,node7
#SBATCH --gres=gpu:a6000:2  # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 96시간 실행

NUM_DEMOS=$1
NUM_ROLLOUTS=$2
STEPS=$3
NUM_GPUS=${4:-2}
BATCH_SIZE=${5:-16}
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE))

BASE_PATH=/home/changyeon
source /home/changyeon/miniconda3/bin/activate gr00t
cd /home/changyeon/workspace/Isaac-GR00T

TASK_NAMES=(
    "peach_diverse"
    "milka_diverse"
    "hichew_diverse"
)

dataset_path=""
for TASK_NAME in ${TASK_NAMES[@]}; do
    dataset_path="${dataset_path} ${BASE_PATH}/data/franka_dataset/diverse_num5_rollouts/${TASK_NAME}_num${NUM_ROLLOUTS}/"
    dataset_path="${dataset_path} ${BASE_PATH}/data/franka_dataset/diverse_num5/${TASK_NAME}_num${NUM_DEMOS}/"
done

script="
    WANDB_PROJECT=droid_panda_wrist_gripper_mt_num5_diverse \
    python scripts/gr00t_finetune.py \
        --dataset-path ${dataset_path} \
        --num-gpus ${NUM_GPUS} \
        --output-dir ${BASE_PATH}/franka_ckpts/diverse_num5/ALL-FT_bs${TOTAL_BATCH_SIZE}_step${STEPS}_demo${NUM_DEMOS}_rollout${NUM_ROLLOUTS} \
        --max-steps ${STEPS} \
        --data-config droid_panda_wrist_gripper \
        --batch-size ${BATCH_SIZE} \
        --save-steps 10000 \
        --run-name ALL-FT_bs${TOTAL_BATCH_SIZE}_step${STEPS}_demo${NUM_DEMOS}_rollout${NUM_ROLLOUTS} \
        --video_backend torchvision_av
"

echo $script
eval $script