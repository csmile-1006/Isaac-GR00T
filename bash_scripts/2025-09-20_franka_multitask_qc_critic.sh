#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=critic_grn15_frk_qc_mt
#SBATCH --output=/home/changyeon/slurm-logs/critic_grn15_frk_qc_mt/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/critic_grn15_frk_qc_mt/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --exclude=node5,node7
#SBATCH --gres=gpu:a6000:2   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=32G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 96시간 실행


NUM_DEMOS=$1
NUM_ROLLOUTS=$2
ACTOR_CKPT_PATH=$3
STEPS=$4
CRITIC_ACTION_HORIZON=$5

DISCOUNT1=${6:-0.995}
DISCOUNT2=${7:-0.995}
NUM_GPUS=${8:-2}
BATCH_SIZE=${9:-16}

BASE_PATH=/home/changyeon/
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE))

TASK_NAMES=(
    "peach_center"
    "banana_center"
    "hichew_center"
)

dataset_path=""
for TASK_NAME in ${TASK_NAMES[@]}; do
    dataset_path="${dataset_path} ${BASE_PATH}/data/franka_dataset/franka_rl_data/rollouts/${TASK_NAME}_num${NUM_ROLLOUTS}/"
    dataset_path="${dataset_path} ${BASE_PATH}/data/franka_dataset/franka_rl_data/${TASK_NAME}_num${NUM_DEMOS}/"
done

source ${BASE_PATH}/miniconda3/bin/activate gr00t
cd ${BASE_PATH}/workspace/Isaac-GR00T
RUN_NAME=QC_Critic_as${CRITIC_ACTION_HORIZON}_bs${TOTAL_BATCH_SIZE}_steps${STEPS}_dm${NUM_DEMOS}_roll${NUM_ROLLOUTS}
python scripts/gr00t_fql_finetune.py \
    --base-model-path ${ACTOR_CKPT_PATH} \
    --dataset-path ${dataset_path} \
    --num-gpus ${NUM_GPUS} \
    --output-dir ${BASE_PATH}/franka_ckpts/multiple/${RUN_NAME}/ \
    --max-steps ${STEPS} \
    --data-config droid_panda_wrist_gripper_rl \
    --batch-size ${BATCH_SIZE} \
    --save-steps 5000 \
    --run-name ${RUN_NAME} \
    --critic-action-horizon ${CRITIC_ACTION_HORIZON} \
    --discount1 ${DISCOUNT1} \
    --discount2 ${DISCOUNT2} \
    --video_backend torchvision_av
