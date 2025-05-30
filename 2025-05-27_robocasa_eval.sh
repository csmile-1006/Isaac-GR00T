#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00t_robocasa_single_ft                                                                                                                                 
#SBATCH --output=/home/visitor_jw/changyeon/slurm-logs/gr00t_robocasa_eval-%j.out  # log                                                                                                   
#SBATCH --error=/home/visitor_jw/changyeon/slurm-logs/gr00t_robocasa_eval-%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:1   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=4     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G    # GPU당 mem 사용량                                                                                                              
#SBATCH --time=4:00:00      # 최대 48시간 실행

TASK_NAME=$1
MODEL_PATH=$2
MODEL_KEY=$3
ACTION_HORIZON=$4
source /home/visitor_jw/anaconda3/bin/activate gr00t
cd /home/visitor_jw/changyeon/Isaac-GR00T

python scripts/eval_policy_robocasa.py \
    --host localhost \
    --port 5555 \
    --data_config single_panda_gripper \
    --action_horizon ${ACTION_HORIZON} \
    --video_backend decord \
    --dataset_path ~/gr00t_dataset/single_panda_gripper.${TASK_NAME}/ \
    --embodiment_tag new_embodiment \
    --model_path ${MODEL_PATH} \
    --env_name ${TASK_NAME} \
    --num_episodes 100 \
    --max_episode_steps 500 \
    --video_path ./test_video/${TASK_NAME}_${MODEL_KEY}_as${ACTION_HORIZON}
