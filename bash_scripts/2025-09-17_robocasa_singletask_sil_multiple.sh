#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=sil_gr00tn15_rc_multiple
#SBATCH --output=/home/changyeon/slurm-logs/sil_gr00tn15_rc_multiple/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/sil_gr00tn15_rc_multiple/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --exclude=node5,node7
#SBATCH --gres=gpu:a6000:2   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=32G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 96시간 실행

NUM_DEMOS=$1
STEPS=$2
NUM_GPUS=${3:-2}
BATCH_SIZE=${4:-16}
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE))

BASE_PATH=/home/changyeon
CKPT_PATH=${BASE_PATH}/ckpts/gr00tn15_rbcs100_bs32_60k
source ${BASE_PATH}/miniconda3/bin/activate gr00t
cd ${BASE_PATH}/workspace/Isaac-GR00T

TASK_NAMES=(
    "CoffeeSetupMug,27",
    "PnPCabToCounter,94",
    "PnPMicrowaveToCounter,73",
    "TurnOffStove,47",
    "PnPCounterToMicrowave,35",
)

dataset_path=""
for TASK in ${TASK_NAMES[@]}; do
    TASK_NAME=$(echo $TASK | cut -d ',' -f 1)
    NUM_ROLLOUT_DEMOS=$(echo $TASK | cut -d ',' -f 2)
    dataset_path="${dataset_path} ${BASE_PATH}/data/robocasa_dataset/${TASK_NAME}_num${NUM_DEMOS}/ ${BASE_PATH}/gr00tn15_robocasa/rollout_demos_num300/${TASK_NAME}_num${NUM_ROLLOUT_DEMOS}/lerobot/"
done


SCRIPT="
    WANDB_PROJECT=gr00t-sil-finetune-multiple \
    python scripts/gr00t_finetune.py \
    --base_model_path ${CKPT_PATH} \
    --dataset-path "${dataset_path}" \
    --num-gpus ${NUM_GPUS} \
    --output-dir "${BASE_PATH}/ckpts/multiple/SIL_bs${TOTAL_BATCH_SIZE}_step${STEPS}_demo${NUM_DEMOS}" \
    --max-steps "${STEPS}" \
    --data-config single_panda_gripper \
    --batch-size ${BATCH_SIZE} \
    --save-steps 10000
"
echo $SCRIPT
eval $SCRIPT