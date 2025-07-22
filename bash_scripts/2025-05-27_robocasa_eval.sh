#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00t_robocasa_single_ft                                                                                                                                 
#SBATCH --output=/home/changyeon/slurm-logs/gr00t_robocasa_eval-%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/gr00t_robocasa_eval-%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:1   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=4     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G    # GPU당 mem 사용량                                                                                                              
#SBATCH --time=4:00:00      # 최대 48시간 실행

NUM_DEMOS=$1
TASK_NAME=$2
MODEL_PATH=$3
MODEL_KEY=$4
ACTION_HORIZON=$5
source /home/changyeon/miniconda3/bin/activate gr00t
cd /home/changyeon/workspace/Isaac-GR00T

python scripts/eval_policy_robocasa.py \
    --host localhost \
    --port 5555 \
    --data_config single_panda_gripper \
    --action_horizon ${ACTION_HORIZON} \
    --video_backend decord \
    --dataset_path ~/robocasa_dataset/mix${NUM_DEMOS}/ \
    --embodiment_tag new_embodiment \
    --model_path ${MODEL_PATH} \
    --env_name ${TASK_NAME} \
    --num_episodes 100 \
    --max_episode_steps 500 \
    --video_path ./test_video/${TASK_NAME}_${MODEL_KEY}_as${ACTION_HORIZON}
