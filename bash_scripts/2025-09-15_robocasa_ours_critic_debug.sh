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
RUN_NAME=debug_OURS_Critic_as${CRITIC_ACTION_HORIZON}_e${EXPECTILE}_d1${DISCOUNT1}_d2${DISCOUNT2}_bs${TOTAL_BATCH_SIZE}_steps${RUN_NAME}_dm${NUM_DEMOS}_roll${NUM_ROLLOUTS}
python scripts/gr00t_ours_critic_finetune.py \
    --dataset-path ${BASE_PATH}/data/robocasa_dataset/${TASK_NAME}_num${NUM_DEMOS}/ ${BASE_PATH}/gr00tn15_robocasa/rollouts/${TASK_NAME}_num${NUM_ROLLOUTS}/lerobot \
    --base-model-path ${CKPT_PATH} \
    --num-gpus ${NUM_GPUS} \
    --output-dir ~/debugs/ckpts/${TASK_NAME}/${RUN_NAME}/ \
    --max-steps ${STEPS} \
    --data-config single_panda_gripper_rl \
    --batch-size ${BATCH_SIZE} \
    --save-steps 50 \
    --run-name debug_${RUN_NAME} \
    --critic-action-horizon ${CRITIC_ACTION_HORIZON} \
    --discount1 ${DISCOUNT1} \
    --discount2 ${DISCOUNT2} \
    --expectile ${EXPECTILE} \

    # --lora_rank 64 \
    # --lora_alpha 128
