#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=mt_eval_gr00tn15
#SBATCH --output=/home/changyeon/slurm-logs/mt_eval_gr00tn15/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/mt_eval_gr00tn15/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:a6000:1   # GPU 1개 사용                                                                                                                  
#SBATCH --array=0-14
#SBATCH --cpus-per-gpu=16     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=128G    # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 48시간 실행

TASK_NAMES=(
    "CoffeeSetupMug,0"
    "PnPCabToCounter,0"
    "PnPMicrowaveToCounter,0"
    "TurnOffStove,0"
    "PnPCounterToMicrowave,0"
    "CoffeeSetupMug,42"
    "PnPCabToCounter,42"
    "PnPMicrowaveToCounter,42"
    "TurnOffStove,42"
    "PnPCounterToMicrowave,42"
    "CoffeeSetupMug,123"
    "PnPCabToCounter,123"
    "PnPMicrowaveToCounter,123"
    "TurnOffStove,123"
    "PnPCounterToMicrowave,123"
)
TASK_NAME=${TASK_NAMES[$SLURM_ARRAY_TASK_ID]%,*}
SEED=${TASK_NAMES[$SLURM_ARRAY_TASK_ID]#*,}
ACTION_HORIZON=16
CKPT_PATH=$1
NUM_ENVS=$2
NUM_ROLLOUTS=$3
SERVER=${4:-"alin_slurm"}
MODEL_TYPE=${5:-"original"}

if [ "$SERVER" == "rlwrld" ]; then
    ROOT_PATH=/virtual_lab/sjw_alinlab/changyeon/
else
    ROOT_PATH=/home/changyeon/
fi

source ${ROOT_PATH}/miniconda3/bin/activate gr00t
cd ${ROOT_PATH}/workspace/Isaac-GR00T

CKPT_FOLDER=$(basename "$(dirname "${CKPT_PATH}")")
CKPT_STEP=$(basename "${CKPT_PATH}")
OUTPUT_PATH=${ROOT_PATH}/gr00tn15_robocasa/evaluations/multiple/${CKPT_FOLDER}/${CKPT_STEP}_s${SEED}_${TASK_NAME}_eval_n${NUM_ROLLOUTS}
script="
    MUJOCO_GL=egl \
    python scripts/eval_policy_robocasa.py \
    --host localhost \
    --port 5555 \
    --model_type ${MODEL_TYPE} \
    --data_config single_panda_gripper \
    --action_horizon ${ACTION_HORIZON} \
    --embodiment_tag new_embodiment \
    --model_path ${CKPT_PATH} \
    --env_name ${TASK_NAME} \
    --num_episodes ${NUM_ROLLOUTS} \
    --noise 0.0 \
    --n_envs ${NUM_ENVS} \
    --output_path ${OUTPUT_PATH} \
    --seed ${SEED} \
    --save_video
"
echo $script
eval $script