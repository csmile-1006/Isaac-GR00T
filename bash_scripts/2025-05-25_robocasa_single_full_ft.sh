#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00t_robocasa_single_ft                                                                                                                                 
#SBATCH --output=/home/changyeon/slurm-logs/gr00t_robocasa_single_ft-%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/gr00t_robocasa_single_ft-%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:a6000:8   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=48:00:00      # 최대 48시간 실행

NUM_DEMOS=$1
STEPS=$2

source /home/changyeon/miniconda3/bin/activate gr00t
cd /home/changyeon/workspace/Isaac-GR00T

python scripts/gr00t_finetune.py \
    --dataset-path ~/robocasa_dataset/mix${NUM_DEMOS}/ \
    --num-gpus 8 \
    --output-dir ~/gr00t_ckpt/step${STEPS}_mix${NUM_DEMOS}_single_panda_gripper \
    --max-steps ${STEPS} \
    --data-config single_panda_gripper \
    --batch-size 12 \
    --save-steps 5000 \
    --resume
