#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00t_n1_5_robocasa_rollout                                                                                                                                 
#SBATCH --output=/home/changyeon/slurm-logs/gr00t_n1_5_robocasa_rollout/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/gr00t_n1_5_robocasa_rollout/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --exclude=node5
#SBATCH --gres=gpu:1   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=4     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G    # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 48시간 실행

TASK_NAME=$1
NUM_DEMOS=$2
STEPS=$3
CHECKPOINT=$4
MODEL_KEY=$5
ACTION_HORIZON=$6
NUM_ROLLOUTS=$7

echo "TASK_NAME: ${TASK_NAME}"
echo "STEPS: ${STEPS}"
echo "CHECKPOINT: ${CHECKPOINT}"
echo "MODEL_KEY: ${MODEL_KEY}"
echo "ACTION_HORIZON: ${ACTION_HORIZON}"
echo "NUM_ROLLOUTS: ${NUM_ROLLOUTS}"

source /home/changyeon/miniconda3/bin/activate gr00t
cd /home/changyeon/workspace/Isaac-GR00T

BASE_PATH=/home/changyeon/evaluation/gr00t_n1_5/${MODEL_KEY}/${TASK_NAME}_num${NUM_ROLLOUTS}_as${ACTION_HORIZON}

python scripts/eval_policy_robocasa.py \
    --host localhost \
    --port 5555 \
    --data_config single_panda_gripper \
    --action_horizon ${ACTION_HORIZON} \
    --embodiment_tag new_embodiment \
    --model_path ~/gr00t_n1_5_ckpt/step${STEPS}_${TASK_NAME}_num${NUM_DEMOS}_single_panda_gripper/checkpoint-${CHECKPOINT} \
    --env_name ${TASK_NAME} \
    --num_episodes ${NUM_ROLLOUTS} \
    --video_path ${BASE_PATH}/videos \
    --collect_data=True \
    --data_collection_path ${BASE_PATH}/data \
    --generative_textures

cd /home/changyeon/workspace/robocasa
OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python robocasa/scripts/dataset_states_to_obs_single.py \
    --dataset ${BASE_PATH}/data/demo.hdf5 \
    --camera_width 256 \
    --camera_height 256 \
    --generative_textures \
    --randomize_cameras \
    --shaped --copy_rewards --copy_dones

cd /home/changyeon/workspace/Isaac-GR00T
python scripts/convert_hdf5_to_lerobot.py \
    --src_hdf5_path ${BASE_PATH}/data/demo_gentex_im256_randcams.hdf5 \
    --output_path ${BASE_PATH}/lerobot \
    --task_name $TASK_NAME \
    --chunks_size 300 \
    --num_episodes ${NUM_ROLLOUTS} \
    --meta_path /home/changyeon/robocasa_dataset/${TASK_NAME}_num${NUM_DEMOS}/meta/
