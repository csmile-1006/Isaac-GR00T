#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00t_robocasa_multitask_eval                                                                                                                                 
#SBATCH --output=/home/changyeon/slurm-logs/gr00t_robocasa_multitask_eval-%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/gr00t_robocasa_multitask_eval-%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:1   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G    # GPU당 mem 사용량                                                                                                              
#SBATCH --time=48:00:00      # 최대 48시간 실행

CHECKPOINT=$1
MODEL_KEY="hw_bs32_2gpu"
MODEL_PATH="~/hw_gr00t_ckpt/checkpoint-${CHECKPOINT}"

echo "MODEL_PATH: ${MODEL_PATH}"
echo "MODEL_KEY: ${MODEL_KEY}"

source /home/changyeon/miniconda3/bin/activate gr00t
cd /home/changyeon/workspace/Isaac-GR00T

TASK_NAMES=(
  "TurnSinkSpout"
  "TurnOnStove"
  "TurnOnSinkFaucet"
  "TurnOnMicrowave"
  "TurnOffStove"
  "TurnOffSinkFaucet"
  "TurnOffMicrowave"
  "PnPStoveToCounter"
  "PnPSinkToCounter"
  "PnPMicrowaveToCounter"
  "PnPCounterToStove"
  "PnPCounterToSink"
  "PnPCounterToMicrowave"
  "PnPCounterToCab"
  "PnPCabToCounter"
  "OpenSingleDoor"
  "OpenDrawer"
  "OpenDoubleDoor"
  "CoffeeSetupMug"
  "CoffeeServeMug"
  "CoffeePressButton"
  "CloseSingleDoor"
  "CloseDrawer"
  "CloseDoubleDoor"
)

for TASK_NAME in ${TASK_NAMES[@]}; do
    CMD="python scripts/eval_policy_robocasa.py \
        --host localhost \
        --port 5555 \
        --data_config single_panda_gripper \
        --embodiment_tag new_embodiment \
        --model_path ${MODEL_PATH} \
        --env_name ${TASK_NAME} \
        --num_episodes 100 \
        --video_path ./evaluation/multitask/${TASK_NAME}/${MODEL_KEY}/videos \
        --generative_textures
    "
    echo $CMD
    eval $CMD
done