#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=rc_grn15_eval_batch
#SBATCH --output=/home/changyeon/slurm-logs/rc_grn15_eval_batch/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/rc_grn15_eval_batch/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gpus=1   # GPU 1개 사용                                                                                                                  
#SBATCH --array=0-8
#SBATCH --cpus-per-gpu=16     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=128G    # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 48시간 실행

TASK_NAME=$1
ACTION_HORIZON=16
CKPT_PATH=$2
NUM_ENVS=$3
NUM_ROLLOUTS=$4
SERVER=${5:-"alin_slurm"}
MODEL_TYPE=${6:-"original"}
TRIAL=${SLURM_ARRAY_TASK_ID}

if [ "$SERVER" == "rlwrld" ]; then
    ROOT_PATH=/virtual_lab/sjw_alinlab/changyeon/
else
    ROOT_PATH=/home/changyeon/
fi

source ${ROOT_PATH}/miniconda3/bin/activate gr00t
cd ${ROOT_PATH}/workspace/Isaac-GR00T

CKPT_FOLDER=$(basename "$(dirname "${CKPT_PATH}")")
CKPT_STEP=$(basename "${CKPT_PATH}")
OUTPUT_PATH=${ROOT_PATH}/gr00tn15_robocasa/evaluations/${TASK_NAME}/${CKPT_FOLDER}/${CKPT_STEP}_eval_n${NUM_ROLLOUTS}_trial${TRIAL}
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
    --save_video
"
echo $script
eval $script