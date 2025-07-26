#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00t_libero_taskwise                                                                                                                                 
#SBATCH --output=/home/changyeon/slurm-logs/libero_taskwise/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/libero_taskwise/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 96시간 실행

TASK_NAME=$1
BATCH_SIZE=$2
NUM_GPUS=$3
STEPS=$4

#SBATCH --gres=gpu:a6000:${NUM_GPUS}

source /home/changyeon/miniconda3/bin/activate gr00t
cd /home/changyeon/workspace/Isaac-GR00T

python scripts/gr00t_finetune.py \
    --dataset-path ~/libero_dataset/libero_gr00t_delta_split/${TASK_NAME}/ \
    --num-gpus ${NUM_GPUS} \
    --output-dir ~/gr00t_n1_5_ckpt/libero/${TASK_NAME}/step${STEPS} \
    --max-steps ${STEPS} \
    --data-config libero \
    --batch-size ${BATCH_SIZE} \
    --save-steps 10000 \
    --run-name GR00T-N1.5-libero-fromPT-step${STEPS}-bs$(($BATCH_SIZE * $NUM_GPUS)) \
