#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=allft_gr00tn15_rc_24
#SBATCH --output=/home/changyeon/slurm-logs/allft_gr00tn15_rc_24/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/allft_gr00tn15_rc_24/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --exclude=node5,node7
#SBATCH --gres=gpu:a6000:2   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=32G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 96시간 실행

NUM_DEMOS=$1
NUM_ROLLOUTS=$2
STEPS=$3
NUM_GPUS=${4:-2}
BATCH_SIZE=${5:-16}
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE))

BASE_PATH=/home/changyeon
CKPT_PATH=${BASE_PATH}/ckpts/gr00tn15_rbcs100_bs32_60k
source ${BASE_PATH}/miniconda3/bin/activate gr00t
cd ${BASE_PATH}/workspace/Isaac-GR00T

TASK_NAMES=(
  "TurnSinkSpout"
  "TurnOnStove"
  "TurnOnSinkFaucet"
  "TurnOnMicrowave"
  "TurnOffStove"
  "TurnOffSinkFaucet"
  "TurnOffMicrowave"
  "PnPStoveToCounter"
  "PnPSinkToCounter"
  "PnPMicrowaveToCounter"
  "PnPCounterToStove"
  "PnPCounterToSink"
  "PnPCounterToMicrowave"
  "PnPCounterToCab"
  "PnPCabToCounter"
  "OpenSingleDoor"
  "OpenDrawer"
  "OpenDoubleDoor"
  "CoffeeSetupMug"
  "CoffeeServeMug"
  "CoffeePressButton"
  "CloseSingleDoor"
  "CloseDrawer"
  "CloseDoubleDoor"
)

dataset_path=""
for TASK in ${TASK_NAMES[@]}; do
    TASK_NAME=$TASK
    dataset_path="${dataset_path} ${BASE_PATH}/data/robocasa_dataset/${TASK_NAME}_num${NUM_DEMOS}/ ${BASE_PATH}/gr00tn15_robocasa/rollouts_num${NUM_ROLLOUTS}/${TASK_NAME}/lerobot/"
done


SCRIPT="
    WANDB_PROJECT=gr00t-all-ft-finetune-24 \
    python scripts/gr00t_finetune.py \
    --base_model_path ${CKPT_PATH} \
    --dataset-path "${dataset_path}" \
    --num-gpus ${NUM_GPUS} \
    --output-dir "${BASE_PATH}/ckpts/24task/ALL-FT_bs${TOTAL_BATCH_SIZE}_step${STEPS}_demo${NUM_DEMOS}_rollout${NUM_ROLLOUTS}" \
    --max-steps "${STEPS}" \
    --data-config single_panda_gripper \
    --batch-size ${BATCH_SIZE} \
    --save-steps 10000 \
    --run-name "ALL-FT_bs${TOTAL_BATCH_SIZE}_step${STEPS}_demo${NUM_DEMOS}_rollout${NUM_ROLLOUTS}" 
"
echo $SCRIPT
eval $SCRIPT