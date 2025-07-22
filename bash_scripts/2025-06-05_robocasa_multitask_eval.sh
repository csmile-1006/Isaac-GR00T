#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=gr00t_robocasa_multitask_eval                                                                                                                                 
#SBATCH --output=/home/changyeon/slurm-logs/gr00t_robocasa_multitask_eval-%j.out  # log                                                                                                   
#SBATCH --error=/home/changyeon/slurm-logs/gr00t_robocasa_multitask_eval-%j.err   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:1   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=48G    # GPU당 mem 사용량                                                                                                              
#SBATCH --time=48:00:00      # 최대 48시간 실행

NUM_DEMOS=$1
STEPS=$2
CHECKPOINT=$3
MODEL_KEY=$4
ACTION_HORIZON=$5

echo "NUM_DEMOS: ${NUM_DEMOS}"
echo "STEPS: ${STEPS}"
echo "CHECKPOINT: ${CHECKPOINT}"
echo "MODEL_KEY: ${MODEL_KEY}"
echo "ACTION_HORIZON: ${ACTION_HORIZON}"

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
        --action_horizon ${ACTION_HORIZON} \
        --video_backend decord \
        --dataset_path ~/robocasa_dataset/mix${NUM_DEMOS}/ \
        --embodiment_tag new_embodiment \
        --model_path ~/gr00t_ckpt/step${STEPS}_mix${NUM_DEMOS}_single_panda_gripper/checkpoint-${CHECKPOINT} \
        --env_name ${TASK_NAME} \
        --num_episodes 100 \
        --max_episode_steps 500 \
        --video_path ./evaluation/multitask/${TASK_NAME}/${MODEL_KEY}_as${ACTION_HORIZON}/videos \
        --data_collection_path ./evaluation/multitask/${TASK_NAME}/${MODEL_KEY}_as${ACTION_HORIZON}/data \
        --generative_textures
    "
    echo $CMD
    eval $CMD
done