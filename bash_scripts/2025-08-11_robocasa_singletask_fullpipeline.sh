#!/bin/bash                                                                                                                                                  
#SBATCH --comment="GR00T-N1.5-3B rollout for offline RL."
#SBATCH --job-name=gr00t_n1_5_robocasa_rollout                                                                                                                                 
#SBATCH --output=/virtual_lab/sjw_alinlab/changyeon/slurm-logs/grn15_rc_rollout/%j.out  # log                                                                                                   
#SBATCH --error=/virtual_lab/sjw_alinlab/changyeon/slurm-logs/grn15_rc_rollout/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gpus=1   # GPU 1개 사용                                                                                                                  
#SBATCH --partition=batch
#SBATCH --array=0-9

TASK_NAMES=(
    "CoffeeSetupMug"
    "CoffeeServeMug"
    "CoffeePressButton"
    "TurnSinkSpout"
    "TurnOnStove"
    "TurnOnSinkFaucet"
    "TurnOnMicrowave"
    "TurnOffStove"
    "TurnOffSinkFaucet"
    "TurnOffMicrowave"
)

TASK_NAME=${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}

ACTION_HORIZON=16
NUM_ENVS=$1
NUM_ROLLOUTS=$2

ROOT_PATH=/virtual_lab/sjw_alinlab/changyeon/

source ${ROOT_PATH}/miniconda3/bin/activate gr00t
cd ${ROOT_PATH}/workspace/Isaac-GR00T

BASE_PATH=${ROOT_PATH}/rollouts/gr00tn15_robocasa/

python scripts/eval_policy_robocasa.py \
    --host localhost \
    --port 5555 \
    --data_config single_panda_gripper \
    --action_horizon ${ACTION_HORIZON} \
    --embodiment_tag new_embodiment \
    --model_path /virtual_lab/sjw_alinlab/changyeon/ckpts/gr00tn15_robocasa/checkpoint-60000/ \
    --env_name ${TASK_NAME} \
    --num_episodes ${NUM_ROLLOUTS} \
    --video_path ${BASE_PATH}/${TASK_NAME}/videos \
    --collect_data \
    --reward_shaping \
    --data_collection_path ${BASE_PATH}/${TASK_NAME}/data \
    --n_envs ${NUM_ENVS}

cd /virtual_lab/sjw_alinlab/changyeon/workspace/robocasa
OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python robocasa/scripts/dataset_states_to_obs_single.py \
    --dataset ${BASE_PATH}/${TASK_NAME}/data/demo.hdf5 \
    --camera_width 256 \
    --camera_height 256 \
    --generative_textures \
    --randomize_cameras \
    --shaped --copy_rewards --copy_dones

cd /virtual_lab/sjw_alinlab/changyeon/workspace/Isaac-GR00T
python scripts/convert_hdf5_to_lerobot.py \
    --src_hdf5_path ${BASE_PATH}/${TASK_NAME}/data/demo_gentex_im256_randcams.hdf5 \
    --output_path ${BASE_PATH}/${TASK_NAME}/lerobot \
    --task_name ${TASK_NAME} \
    --chunks_size 1000 \
    --num_episodes ${NUM_ROLLOUTS} \
    --meta_path ${ROOT_PATH}/data/robocasa_dataset/single_panda_gripper.${TASK_NAME}/meta/
