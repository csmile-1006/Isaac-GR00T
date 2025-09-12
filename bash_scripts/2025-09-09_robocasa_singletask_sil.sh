#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=sil_gr00tn15_rc_single
#SBATCH --output=/home/changyeon/slurm-logs/sil_gr00tn15_rc_single/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/sil_gr00tn15_rc_single/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --exclude=node5,node7
#SBATCH --gres=gpu:a6000:2   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=32G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 96시간 실행

TASK_NAME=$1
NUM_DEMOS=$2
NUM_ROLLOUTS=$3
STEPS=$4
NUM_GPUS=${5:-2}
BATCH_SIZE=${6:-16}
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE))

BASE_PATH=/home/changyeon
CKPT_PATH=${BASE_PATH}/ckpts/gr00tn15_rbcs_bs32_60k
source ${BASE_PATH}/miniconda3/bin/activate gr00t
cd ${BASE_PATH}/workspace/Isaac-GR00T

SCRIPT="
    WANDB_PROJECT=gr00t-sil-finetune \
    python scripts/gr00t_finetune.py \
    --base_model_path ${CKPT_PATH} \
    --dataset-path "${BASE_PATH}/data/robocasa_dataset/${TASK_NAME}_num${NUM_DEMOS}/" \
                   "${BASE_PATH}/gr00tn15_robocasa/rollout_demos/${TASK_NAME}_num${NUM_ROLLOUTS}/lerobot/" \
    --num-gpus ${NUM_GPUS} \
    --output-dir "${BASE_PATH}/ckpts/${TASK_NAME}/SIL_bs${TOTAL_BATCH_SIZE}_step${STEPS}_demo${NUM_DEMOS}" \
    --max-steps "${STEPS}" \
    --data-config single_panda_gripper \
    --batch-size ${BATCH_SIZE} \
    --save-steps 5000
"
echo $SCRIPT
eval $SCRIPT