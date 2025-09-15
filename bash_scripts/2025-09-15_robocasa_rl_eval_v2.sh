#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00tn15_rc_rollouts
#SBATCH --output=/home/changyeon/slurm-logs/gr00tn15_rc_rollouts/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/gr00tn15_rc_rollouts/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gpus=1   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=16     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=128G    # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 48시간 실행

TASK_NAME=$1
ACTOR_CKPT_PATH=$2
CRITIC_CKPT_PATH=$3
NUM_ENVS=$4
NUM_ROLLOUTS=$5
SERVER=${6:-"alin_slurm"}
MODEL_TYPE=${7:-"ours_dual_bon"}
ACTION_HORIZON=${8:-16}
NUM_SAMPLES=${9:-4}

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
    --save_video \
    --num_samples ${NUM_SAMPLES}
"
echo $script
eval $script