#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=eval_grn15_rc_rl_v2_batch
#SBATCH --output=/home/changyeon/slurm-logs/eval_grn15_rc_rl_v2_batch/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/eval_grn15_rc_rl_v2_batch/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gpus=1   # GPU 1개 사용                                                                                                                  
#SBATCH --array=0-11
#SBATCH --cpus-per-gpu=8    # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=16G    # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 48시간 실행

HYPERPARAMS=(
    "10000,4"
    "10000,8"
    "10000,16"
    "20000,4"
    "20000,8"
    "20000,16"
    "30000,4"
    "30000,8"
    "30000,16"
    "40000,4"
    "40000,8"
    "40000,16"
)
IFS=',' read STEPS NUM_SAMPLES <<< "${HYPERPARAMS[$SLURM_ARRAY_TASK_ID]}"

TASK_NAME=$1
ACTOR_CKPT_PATH=$2
CRITIC_CKPT_PATH=$3
NUM_ENVS=$4
NUM_ROLLOUTS=$5
SERVER=${6:-"alin_slurm"}
MODEL_TYPE=${7:-"ours_dual_bon"}
ACTION_HORIZON=${8:-16}

CRITIC_CKPT_PATH=${CRITIC_CKPT_PATH}/step-${STEPS}

if [ "$SERVER" == "rlwrld" ]; then
    ROOT_PATH=/virtual_lab/sjw_alinlab/changyeon/
else
    ROOT_PATH=/home/changyeon/
fi

source ${ROOT_PATH}/miniconda3/bin/activate gr00t
cd ${ROOT_PATH}/workspace/Isaac-GR00T

CKPT_FOLDER=$(basename "$(dirname "${CRITIC_CKPT_PATH}")")
CKPT_STEP=$(basename "${CRITIC_CKPT_PATH}")
OUTPUT_PATH=${ROOT_PATH}/gr00tn15_robocasa/evaluations/${TASK_NAME}/${CKPT_FOLDER}/${CKPT_STEP}_eval_n${NUM_ROLLOUTS}_bo${NUM_SAMPLES}
script="
    MUJOCO_GL=egl \
    python scripts/eval_policy_robocasa_v2.py \
    --host localhost \
    --port 5555 \
    --model_type ${MODEL_TYPE} \
    --data_config single_panda_gripper_rl_inference \
    --action_horizon ${ACTION_HORIZON} \
    --embodiment_tag new_embodiment \
    --actor_model_path ${ACTOR_CKPT_PATH} \
    --critic_model_path ${CRITIC_CKPT_PATH} \
    --env_name ${TASK_NAME} \
    --num_episodes ${NUM_ROLLOUTS} \
    --noise 0.0 \
    --n_envs ${NUM_ENVS} \
    --output_path ${OUTPUT_PATH} \
    --num_samples ${NUM_SAMPLES}
"
echo $script
eval $script