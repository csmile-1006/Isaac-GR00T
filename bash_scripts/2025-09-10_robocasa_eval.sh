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
ACTION_HORIZON=16
CKPT_NAME=$2
CKPT_STEP=$3
NUM_ENVS=$4
NUM_ROLLOUTS=$5
SERVER=${6:-"alin_slurm"}

if [ "$SERVER" == "rlwrld" ]; then
    ROOT_PATH=/virtual_lab/sjw_alinlab/changyeon/
else
    ROOT_PATH=/home/changyeon/
fi

source ${ROOT_PATH}/miniconda3/bin/activate gr00t
cd ${ROOT_PATH}/workspace/Isaac-GR00T

MUJOCO_GL=egl
BASE_PATH=${ROOT_PATH}/gr00tn15_robocasa/rollouts/
CKPT_PATH=${ROOT_PATH}/ckpts/${CKPT_NAME}/checkpoint-${CKPT_STEP}

python scripts/eval_policy_robocasa.py \
    --host localhost \
    --port 5555 \
    --data_config single_panda_gripper \
    --action_horizon ${ACTION_HORIZON} \
    --embodiment_tag new_embodiment \
    --model_path ${CKPT_PATH} \
    --env_name ${TASK_NAME} \
    --num_episodes ${NUM_ROLLOUTS} \
    --noise 0.0 \
    --n_envs ${NUM_ENVS}
