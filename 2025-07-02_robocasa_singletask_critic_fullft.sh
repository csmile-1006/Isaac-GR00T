#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00t_rl_critic
#SBATCH --output=/home/changyeon/slurm-logs/gr00t_rl_critic/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/gr00t_rl_critic/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --exclude=node7
#SBATCH --gres=gpu:a6000:1   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=1     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=48:00:00      # 최대 96시간 실행

TASK_NAME=$1
NUM_DEMOS=$2
NUM_EPISODES=$3
CHECKPOINT=$4
STEPS=$5
TRAINING_STEPS=$6

source /home/changyeon/miniconda3/bin/activate gr00t
cd /home/changyeon/workspace/Isaac-GR00T

python scripts/train_rl_critic.py \
    --dataset-path ~/robocasa_dataset/${TASK_NAME}_num${NUM_DEMOS}/ ~/evaluation/gr00t_n1_5/rl_data/${TASK_NAME}_num${NUM_EPISODES}_as16/lerobot/ \
    --base-model-path ~/gr00t_n1_5_ckpt/step${CHECKPOINT}_${TASK_NAME}_num${NUM_DEMOS}_single_panda_gripper/checkpoint-${CHECKPOINT} \
    --num-gpus 1 \
    --output-dir ~/gr00t_n1_5/critic_ckpt/critic_step${STEPS}_${TASK_NAME}_num${NUM_DEMOS}_single_panda_gripper \
    --max-steps ${TRAINING_STEPS} \
    --data-config single_panda_gripper_state_rl \
    --batch-size 256 \
    --save-steps 100000
