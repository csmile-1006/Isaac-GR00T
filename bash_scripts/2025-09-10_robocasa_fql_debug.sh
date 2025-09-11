TASK_NAME=$1
NUM_DEMOS=$2
NUM_ROLLOUTS=$3
STEPS=$4
CRITIC_ACTION_HORIZON=$5
NUM_GPUS=${6:-2}
BATCH_SIZE=${7:-32}

BASE_PATH=/home/changyeon/
CKPT_PATH=${BASE_PATH}/ckpts/gr00tn15_rbcs100_bs32_60k/
source ${BASE_PATH}/miniconda3/bin/activate gr00t
cd ${BASE_PATH}/workspace/Isaac-GR00T
RUN_NAME=${TASK_NAME}_num${NUM_DEMOS}_num${NUM_ROLLOUTS}_bs$(($BATCH_SIZE * $NUM_GPUS))_step${STEPS}
python scripts/gr00t_fql_finetune.py \
    --dataset-path ${BASE_PATH}/data/robocasa_dataset/${TASK_NAME}_num${NUM_DEMOS}/ ${BASE_PATH}/gr00tn15_robocasa/rl_rollouts/${TASK_NAME}_num${NUM_ROLLOUTS}/lerobot \
    --base-model-path ${CKPT_PATH} \
    --num-gpus ${NUM_GPUS} \
    --output-dir ~/debugs/FQL/${RUN_NAME}/ \
    --max-steps ${STEPS} \
    --data-config single_panda_gripper_rl \
    --batch-size ${BATCH_SIZE} \
    --save-steps 100 \
    --run-name debug_${RUN_NAME} \
    --critic-action-horizon ${CRITIC_ACTION_HORIZON} \
    # --lora_rank 64 \
    # --lora_alpha 128
