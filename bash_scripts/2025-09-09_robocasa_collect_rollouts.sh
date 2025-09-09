#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00tn15_rc_rollouts
#SBATCH --output=/home/changyeon/slurm-logs/gr00tn15_rc_rollouts/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/gr00tn15_rc_rollouts/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gpus=1   # GPU 1개 사용                                                                                                                  
#SBATCH --array=0-0
#SBATCH --cpus-per-gpu=16     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=128G    # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 48시간 실행

TASK_NAMES=(
    # "CoffeeServeMug"
    "CoffeeSetupMug"
    # "CoffeeServeMug"
    # "CoffeePressButton"
    # "TurnSinkSpout"
    # "TurnOnStove"
    # "TurnOnSinkFaucet"
    # "TurnOnMicrowave"
    # "TurnOffStove"
    # "TurnOffSinkFaucet"
    # "TurnOffMicrowave"
    # "PnPStoveToCounter"
    # "PnPSinkToCounter"
    # "PnPMicrowaveToCounter"
    # "PnPCounterToStove"
    # "PnPCounterToSink"
    # "PnPCounterToMicrowave"
    # "PnPCounterToCab"
    # "PnPCabToCounter"
    # "OpenSingleDoor"
    # "OpenDrawer"
    # "OpenDoubleDoor"
    # "CloseSingleDoor"
    # "CloseDrawer"
    # "CloseDoubleDoor"
)

TASK_NAME=${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}

ACTION_HORIZON=16
NUM_ENVS=$1
NUM_ROLLOUTS=$2
SERVER=${3:-"alin_slurm"}
NUM_PROCS=${4:-4}

if [ "$SERVER" == "rlwrld" ]; then
    ROOT_PATH=/virtual_lab/sjw_alinlab/changyeon/
else
    ROOT_PATH=/home/changyeon/
fi

source ${ROOT_PATH}/miniconda3/bin/activate gr00t
cd ${ROOT_PATH}/workspace/Isaac-GR00T

MUJOCO_GL=egl
BASE_PATH=${ROOT_PATH}/gr00tn15_robocasa/rollouts/
CKPT_PATH=${ROOT_PATH}/ckpts/gr00tn15_rbcs_bs32_60k

python scripts/eval_policy_robocasa.py \
    --host localhost \
    --port 5555 \
    --data_config single_panda_gripper \
    --action_horizon ${ACTION_HORIZON} \
    --embodiment_tag new_embodiment \
    --model_path ${CKPT_PATH} \
    --env_name ${TASK_NAME} \
    --num_episodes ${NUM_ROLLOUTS} \
    --collect_data \
    --reward_shaping \
    --noise 0.1 \
    --data_collection_path ${BASE_PATH}/${TASK_NAME}_num${NUM_ROLLOUTS}/data \
    --n_envs ${NUM_ENVS}

cd ${ROOT_PATH}/workspace/robocasa
OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python robocasa/scripts/dataset_states_to_obs_robust.py \
    --dataset ${BASE_PATH}/${TASK_NAME}_num${NUM_ROLLOUTS}/data/demo.hdf5 \
    --camera_width 256 \
    --camera_height 256 \
    --generative_textures \
    --randomize_cameras \
    --shaped --copy_rewards --copy_dones \
    --num_procs ${NUM_PROCS}

cd ${ROOT_PATH}/workspace/Isaac-GR00T
python scripts/convert_hdf5_to_lerobot.py \
    --src_hdf5_path ${BASE_PATH}/${TASK_NAME}_num${NUM_ROLLOUTS}/data/demo_gentex_im256_randcams.hdf5 \
    --output_path ${BASE_PATH}/${TASK_NAME}_num${NUM_ROLLOUTS}/lerobot \
    --task_name ${TASK_NAME} \
    --chunks_size 1000 \
    --num_episodes ${NUM_ROLLOUTS} \
    --num_video_workers ${NUM_PROCS}
