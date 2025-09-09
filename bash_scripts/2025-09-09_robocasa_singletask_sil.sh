#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00tn15_rc_single_sil                                                                                                                                 
#SBATCH --output=/home/changyeon/slurm-logs/gr00tn15_rc_single_sil/%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/gr00tn15_rc_single_sil/%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:a6000:2   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=32G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 96시간 실행

TASK_NAME=$1
NUM_DEMOS=$2
STEPS=$3

BASE_PATH=/home/changyeon/
CKPT_PATH=${BASE_PATH}/ckpts/gr00tn15_rbcs_bs32_60k
source ${BASE_PATH}/miniconda3/bin/activate gr00t
cd ${BASE_PATH}/workspace/Isaac-GR00T

python scripts/gr00t_finetune.py \
    --base_model_path ${CKPT_PATH} \
    --dataset-path "${BASE_PATH}/data/robocasa_dataset/${TASK_NAME}_num${NUM_DEMOS}/" \
                   "${BASE_PATH}/gr00tn15_robocasa/rollout_demos/${TASK_NAME}_num${NUM_DEMOS}/lerobot/" \
    --num-gpus 2 \
    --output-dir "${BASE_PATH}/ckpts/SIL/step${STEPS}_${TASK_NAME}_num${NUM_DEMOS}" \
    --max-steps "${STEPS}" \
    --data-config single_panda_gripper \
    --batch-size 16 \
    --save-steps 5000
