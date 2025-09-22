#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=eval_grn15_rc_qc_mt
#SBATCH --output=/home/changyeon/slurm-logs/eval_grn15_rc_qc_mt/%A_%a.out  # log
#SBATCH --error=/home/changyeon/slurm-logs/eval_grn15_rc_qc_mt/%A_%a.err   # log
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:a6000:1   # GPU 1개 사용                                                                                                                  
#SBATCH --array=0-5%10
#SBATCH --cpus-per-gpu=8    # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=64G    # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 48시간 실행

SEEDS=(
    3407
)
TASK_NAMES=(
    "CoffeeSetupMug"
    "PnPCabToCounter"
    "PnPMicrowaveToCounter"
    "TurnOffStove"
    "PnPCounterToMicrowave"
)

CRITIC_CKPT_PATHS=(
    "/home/changyeon/ckpts/multiple/QC_Critic_as16_bs32_steps30000_dm100_roll300,qc,30000"
    # "/home/changyeon/ckpts/multiple/OURS_Critic_as16_e0.7_d10.9_d20.99_bs64_steps_dm100_roll300,ours_dual_bon,30000"
    # "/home/changyeon/ckpts/multiple/IQL_Critic_as1_e0.7_bs32_steps_dm100_roll300,iql_dual_bon,30000"
)

ACTOR_CKPT_PATH=$1
ACTOR_CKPT_TYPE=$2
NUM_ENVS=$3
NUM_ROLLOUTS=$4
SERVER=${5:-"alin_slurm"}
ACTION_HORIZON=${6:-16}
N_SAMPLES=${7:-10}

# Create a single HYPERPARAMS list that repeats the same setup for each TASK_NAME
HYPERPARAMS=()
for TASK in "${TASK_NAMES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for CKPT_LINE in "${CRITIC_CKPT_PATHS[@]}"; do
            IFS=',' read -r CRITIC_CKPT_PATH MODEL_TYPE CRITIC_CHECKPOINT <<< "$CKPT_LINE"
            HYPERPARAMS+=(
                "${CRITIC_CHECKPOINT},${N_SAMPLES},0,${TASK},${SEED},${CRITIC_CKPT_PATH},${MODEL_TYPE},${CRITIC_CHECKPOINT}"
                # "${CRITIC_CHECKPOINT},${N_SAMPLES},1.0,${TASK},${SEED},${CRITIC_CKPT_PATH},${MODEL_TYPE},${CRITIC_CHECKPOINT}"
                # "${CRITIC_CHECKPOINT},$((N_SAMPLES * 5)),0,${TASK},${SEED},${CRITIC_CKPT_PATH},${MODEL_TYPE},${CRITIC_CHECKPOINT}"
                # "${CRITIC_CHECKPOINT},$((N_SAMPLES * 5)),1.0,${TASK},${SEED},${CRITIC_CKPT_PATH},${MODEL_TYPE},${CRITIC_CHECKPOINT}"
            )
        done
    done
done
IFS=',' read STEPS NUM_SAMPLES TEMPERATURE TASK_NAME SEED CRITIC_CKPT_PATH MODEL_TYPE STEPS <<< "${HYPERPARAMS[$SLURM_ARRAY_TASK_ID]}"


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
OUTPUT_PATH=${ROOT_PATH}/gr00tn15_robocasa/evaluations/multiple/${ACTOR_CKPT_TYPE}_${CKPT_FOLDER}/${CKPT_STEP}_s${SEED}_${TASK_NAME}_eval_n${NUM_ROLLOUTS}_bo${NUM_SAMPLES}_t${TEMPERATURE}
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
    --temperature ${TEMPERATURE} \
    --seed ${SEED}
"
echo $script
eval $script
