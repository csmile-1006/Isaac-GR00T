#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00t_robocasa_single_ft                                                                                                                                 
#SBATCH --output=/home/visitor_jw/changyeon/slurm-logs/gr00t_robocasa_single_ft-%j.out  # log                                                                                                   
#SBATCH --error=/home/visitor_jw/changyeon/slurm-logs/gr00t_robocasa_single_ft-%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:a6000:8   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=4     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=12:00:00      # 최대 48시간 실행

TASK_NAME=$1
source /home/visitor_jw/anaconda3/bin/activate gr00t
cd /home/visitor_jw/changyeon/Isaac-GR00T

python scripts/gr00t_finetune.py \
    --dataset-path ~/gr00t_dataset/single_panda_gripper.${TASK_NAME}/ \
    --num-gpus 8 \
    --output-dir ~/changyeon/gr00t_lora_ckpt/single_panda_gripper.${TASK_NAME} \
    --max-steps 20000 \
    --data-config single_panda_gripper \
    --batch-size 32 \
    --num-demos 300 \
    --save-steps 5000 \
    --lora_rank 64 \
    --lora_alpha 128
