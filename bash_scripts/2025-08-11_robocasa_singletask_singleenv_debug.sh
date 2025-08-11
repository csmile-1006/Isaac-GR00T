#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00t_n1_5_robocasa_rollout                                                                                                                                 
#SBATCH --output=/home/changyeon/debug/slurm-logs/grn15_rc_rollout_singleenv/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/debug/slurm-logs/grn15_rc_rollout_singleenv/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --exclude=node5
#SBATCH --gres=gpu:1   # GPU 1개 사용                                                                                                                  
#SBATCH --array=0-2
#SBATCH --cpus-per-gpu=4     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G    # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 48시간 실행

TASK_NAMES=(
    "CoffeeSetupMug"
    "CoffeeServeMug"
    "CoffeePressButton"
)

TASK_NAME=${TASK_NAMES[0]}

NUM_DEMOS=300
ACTION_HORIZON=16
NUM_ROLLOUTS=$1

source /home/changyeon/miniconda3/bin/activate gr00t
cd /home/changyeon/workspace/Isaac-GR00T

BASE_PATH=/home/changyeon/debug/singleenv/gr00tn15_robocasa/

python scripts/eval_policy_robocasa_singleenv.py \
    --host localhost \
    --port 5555 \
    --data_config single_panda_gripper \
    --action_horizon ${ACTION_HORIZON} \
    --embodiment_tag new_embodiment \
    --model_path /home/huiwon/gr00tn15_robocasa/ \
    --env_name ${TASK_NAME} \
    --num_episodes ${NUM_ROLLOUTS} \
    --video_path ${BASE_PATH}/${TASK_NAME}/videos \
    --collect_data \
    --data_collection_path ${BASE_PATH}/${TASK_NAME}/data \

# cd /home/changyeon/workspace/robocasa
# OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python robocasa/scripts/dataset_states_to_obs_single.py \
#     --dataset ${BASE_PATH}/${TASK_NAME}/data/demo.hdf5 \
#     --camera_width 256 \
#     --camera_height 256 \
#     --generative_textures \
#     --randomize_cameras \
#     --shaped --copy_rewards --copy_dones

# cd /home/changyeon/workspace/Isaac-GR00T
# python scripts/convert_hdf5_to_lerobot.py \
#     --src_hdf5_path ${BASE_PATH}/${TASK_NAME}/data/demo_gentex_im256_randcams.hdf5 \
#     --output_path ${BASE_PATH}/lerobot/${TASK_NAME} \
#     --task_name ${TASK_NAME} \
#     --chunks_size 300 \
#     --num_episodes ${NUM_ROLLOUTS} \
#     --meta_path /home/changyeon/robocasa_dataset/${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}_num${NUM_DEMOS}/meta/
