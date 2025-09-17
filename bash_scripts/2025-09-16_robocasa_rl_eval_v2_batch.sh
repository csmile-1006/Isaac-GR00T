#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=eval_grn15_rc_rl_v2_batch
#SBATCH --output=/home/changyeon/slurm-logs/eval_grn15_rc_rl_v2_batch/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/eval_grn15_rc_rl_v2_batch/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gpus=1   # GPU 1개 사용                                                                                                                  
#SBATCH --array=0-5
#SBATCH --cpus-per-gpu=16    # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=128G    # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 48시간 실행

TASK_NAME=$1
ACTOR_CKPT_PATH=$2
ACTOR_CKPT_TYPE=$3
CRITIC_CKPT_PATH=$4
CRITIC_CHECKPOINT=$5
NUM_ENVS=$6
NUM_ROLLOUTS=$7
SERVER=${8:-"alin_slurm"}
MODEL_TYPE=${9:-"ours_dual_bon"}
ACTION_HORIZON=${10:-16}
N_SAMPLES=${11:-10}

HYPERPARAMS=(
    # "10000,16,0.1"
    # "20000,16,0.1"
    # "30000,16,0.1"
    # "40000,16,0.1"
    # "10000,16,1.0"
    # "20000,16,1.0"
    # "30000,16,1.0"
    # "40000,16,1.0"
    # "30000,${N_SAMPLES},0"
    # "60000,${N_SAMPLES},0"
    # "90000,${N_SAMPLES},0"
    # "30000,${N_SAMPLES},1.0"
    # "60000,${N_SAMPLES},1.0"
    # "90000,${N_SAMPLES},1.0"
    # "30000,${N_SAMPLES},0.1"
    # "60000,${N_SAMPLES},0.1"
    "${CHECKPOINT},${N_SAMPLES},0"
    "${CHECKPOINT},${N_SAMPLES},1.0"
    "${CHECKPOINT},$((N_SAMPLES * 2)),0"
    "${CHECKPOINT},$((N_SAMPLES * 2)),1.0"
    "${CHECKPOINT},$((N_SAMPLES * 5)),0"
    "${CHECKPOINT},$((N_SAMPLES * 5)),1.0"
)

IFS=',' read STEPS NUM_SAMPLES TEMPERATURE <<< "${HYPERPARAMS[$SLURM_ARRAY_TASK_ID]}"


CRITIC_CKPT_PATH=${CRITIC_CKPT_PATH}/checkpoint-${STEPS}

if [ "$SERVER" == "rlwrld" ]; then
    ROOT_PATH=/virtual_lab/sjw_alinlab/changyeon/
else
    ROOT_PATH=/home/changyeon/
fi

source ${ROOT_PATH}/miniconda3/bin/activate gr00t
cd ${ROOT_PATH}/workspace/Isaac-GR00T

CKPT_FOLDER=$(basename "$(dirname "${CRITIC_CKPT_PATH}")")
CKPT_STEP=$(basename "${CRITIC_CKPT_PATH}")
OUTPUT_PATH=${ROOT_PATH}/gr00tn15_robocasa/evaluations/${TASK_NAME}/${ACTOR_CKPT_TYPE}_${CKPT_FOLDER}/${CKPT_STEP}_eval_n${NUM_ROLLOUTS}_bo${NUM_SAMPLES}_t${TEMPERATURE}
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
    --num_samples ${NUM_SAMPLES} \
    --temperature ${TEMPERATURE}
"
echo $script
eval $script