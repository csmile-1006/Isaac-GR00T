#!/bin/bash

# Split TASK_NAMES into 4 sets and select by SET_ID (1, 2, 3, or 4)

CKPT_ID=${1:-"num1000-checkpoint-30000"}
NUM_ENVS=${2:-1}
NUM_EPISODES=${3:-20}

TASK_NAMES=(
    "PosttrainPnPNovelFromCuttingboardToPanSplitA"
    "PosttrainPnPNovelFromCuttingboardToPotSplitA"
    "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA"
    "PosttrainPnPNovelFromPlacematToBasketSplitA"
    "PosttrainPnPNovelFromPlacematToBowlSplitA"
    "PosttrainPnPNovelFromPlacematToPlateSplitA"
    "PosttrainPnPNovelFromPlacematToTieredshelfSplitA"
    "PosttrainPnPNovelFromPlateToBowlSplitA"
)

echo "Running evaluation for TASK: ${TASK_NAMES[*]}"
BASE_PATH="/home/changyeon/evaluations/${CKPT_ID}"
for TASK_NAME in "${TASK_NAMES[@]}"; do
    python3 scripts/simulation_service.py --client \
        --max_episode_steps 720 \
        --n_envs ${NUM_ENVS} \
        --n_episodes ${NUM_EPISODES} \
        --env_name gr1_unified/${TASK_NAME}_GR1ArmsAndWaistFourierHands_Env \
        --output_dir $BASE_PATH/stats/gr1_unified.${TASK_NAME}_GR1ArmsAndWaistFourierHands_Env \
        --video_dir $BASE_PATH/videos/gr1_unified.${TASK_NAME}_GR1ArmsAndWaistFourierHands_Env \
        --data_name ${TASK_NAME}_GR1ArmsAndWaistFourierHands_Env \
        --lerobot_output_dir $BASE_PATH/rollouts/gr1_unified.${TASK_NAME}_GR1ArmsAndWaistFourierHands_Env
done