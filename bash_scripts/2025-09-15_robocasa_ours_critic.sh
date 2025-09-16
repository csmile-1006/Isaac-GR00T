#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=critic_grn15_rc
#SBATCH --output=/home/changyeon/slurm-logs/critic_grn15_rc/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/critic_grn15_rc/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --exclude=node5,node7
#SBATCH --gres=gpu:a6000:2   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=32G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 96시간 실행


TASK_NAME=$1
NUM_DEMOS=$2
NUM_ROLLOUTS=$3
STEPS=$4
CRITIC_ACTION_HORIZON=$5
CKPT_TYPE=${6:-"SIL"}
DISCOUNT1=${7:-0.995}
DISCOUNT2=${8:-0.995}
EXPECTILE=${9:-0.9}
NUM_GPUS=${10:-2}
BATCH_SIZE=${11:-16}

BASE_PATH=/home/changyeon/
if [ ${CKPT_TYPE} == "SIL" ]; then
    CKPT_PATH=${BASE_PATH}/ckpts/${TASK_NAME}/SIL_bs32_step30000_demo100_rolldemo225/
elif [ ${CKPT_TYPE} == "BASE" ]; then
    CKPT_PATH=${BASE_PATH}/ckpts/gr00tn15_rbcs100_bs32_60k/
fi
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE))

source ${BASE_PATH}/miniconda3/bin/activate gr00t
cd ${BASE_PATH}/workspace/Isaac-GR00T
RUN_NAME=OURS_Critic_${CKPT_TYPE}_as${CRITIC_ACTION_HORIZON}_e${EXPECTILE}_d1${DISCOUNT1}_d2${DISCOUNT2}_bs${TOTAL_BATCH_SIZE}_steps${RUN_NAME}_dm${NUM_DEMOS}_roll${NUM_ROLLOUTS}
python scripts/gr00t_ours_critic_finetune.py \
    --dataset-path ${BASE_PATH}/data/robocasa_dataset/${TASK_NAME}_num${NUM_DEMOS}/ ${BASE_PATH}/gr00tn15_robocasa/rollouts/${TASK_NAME}_num${NUM_ROLLOUTS}/lerobot \
    --base-model-path ${CKPT_PATH} \
    --num-gpus ${NUM_GPUS} \
    --output-dir ${BASE_PATH}/ckpts/${TASK_NAME}/${RUN_NAME}/ \
    --max-steps ${STEPS} \
    --data-config single_panda_gripper_rl \
    --batch-size ${BATCH_SIZE} \
    --save-steps 10000 \
    --run-name ${RUN_NAME} \
    --critic-action-horizon ${CRITIC_ACTION_HORIZON} \
    --discount1 ${DISCOUNT1} \
    --discount2 ${DISCOUNT2} \
    --expectile ${EXPECTILE} \

    # --lora_rank 64 \
    # --lora_alpha 128
