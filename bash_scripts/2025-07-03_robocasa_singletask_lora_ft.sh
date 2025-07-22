#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00t_robocasa_single_lora_ft                                                                                                                                 
#SBATCH --output=/home/changyeon/slurm-logs/gr00t_robocasa_single_lora_ft/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/gr00t_robocasa_single_lora_ft/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:a6000:4   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=4     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 48시간 실행

TASK_NAME=$1
NUM_DEMOS=$2
NUM_EPISODES=$3

source /home/changyeon/miniconda3/bin/activate gr00t
cd /home/changyeon/workspace/Isaac-GR00T

python scripts/gr00t_finetune.py \
    --dataset-path ~/robocasa_dataset/${TASK_NAME}_num${NUM_DEMOS}/ ~/evaluation/gr00t_n1_5/rl_test/${TASK_NAME}_num${NUM_EPISODES}_as16/ \
    --num-gpus 4 \
    --output-dir ~/gr00t_n1_5_ckpt/lora/step${STEPS}_${TASK_NAME}_num${NUM_DEMOS}_single_panda_gripper \
    --max-steps 60000 \
    --data-config single_panda_gripper \
    --batch-size 64 \
    --save-steps 10000 \
    --lora_rank 64 \
    --lora_alpha 128
