#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00t_robocasa_single_ft                                                                                                                                 
#SBATCH --output=/home/visitor_jw/changyeon/slurm-logs/gr00t_robocasa_single_multinode_ft-%j.out  # log                                                                                                   
#SBATCH --error=/home/visitor_jw/changyeon/slurm-logs/gr00t_robocasa_single_multinode_ft-%j.err   # log                                                                                                   
#SBATCH --nodes=2            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:a6000:2   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=4     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=24:00:00      # 최대 48시간 실행

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $2}')
export MASTER_PORT=29500

echo "NODES: $SLURM_JOB_NODELIST"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# 선택적으로 환경 로그 출력
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"

TASK_NAME=$1
source /home/visitor_jw/anaconda3/bin/activate gr00t
cd /home/visitor_jw/changyeon/Isaac-GR00T

export NCCL_DEBUG=INFO

srun python scripts/gr00t_finetune_multinode.py \
    --dataset-path ~/gr00t_dataset/single_panda_gripper.${TASK_NAME}/ \
    --num-gpus 2 \
    --output-dir ~/changyeon/gr00t_multinode_ckpt/single_panda_gripper.${TASK_NAME} \
    --max-steps 20000 \
    --data-config single_panda_gripper \
    --batch-size 12 \
    --num-demos 300 \
    --save-steps 5000 \