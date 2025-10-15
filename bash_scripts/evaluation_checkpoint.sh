#!/bin/bash

# Split TASK_NAMES into 4 sets and select by SET_ID (1, 2, 3, or 4)

SET_ID=${1:-1}  # Usage: bash evaluation_checkpoint.sh [SET_ID], default is 1
CKPT_ID=${2:-"num1000-checkpoint-30000"}

TASK_NAMES_SET_1=(
    "PnPBottleToCabinetClose"
    "PnPCanToDrawerClose"
    "PnPCupToDrawerClose"
    "PnPMilkToMicrowaveClose"
    "PnPPotatoToMicrowaveClose"
    "PnPWineToCabinetClose"
)

TASK_NAMES_SET_2=(
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA"
    "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA"
    "PosttrainPnPNovelFromCuttingboardToPanSplitA"
    "PosttrainPnPNovelFromCuttingboardToPotSplitA"
    "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA"
    "PosttrainPnPNovelFromPlacematToBasketSplitA"
)

TASK_NAMES_SET_3=(
    "PosttrainPnPNovelFromPlacematToBowlSplitA"
    "PosttrainPnPNovelFromPlacematToPlateSplitA"
    "PosttrainPnPNovelFromPlacematToTieredshelfSplitA"
    "PosttrainPnPNovelFromPlateToBowlSplitA"
    "PosttrainPnPNovelFromPlateToCardboardboxSplitA"
    "PosttrainPnPNovelFromPlateToPanSplitA"
)

TASK_NAMES_SET_4=(
    "PosttrainPnPNovelFromPlateToPlateSplitA"
    "PosttrainPnPNovelFromTrayToCardboardboxSplitA"
    "PosttrainPnPNovelFromTrayToPlateSplitA"
    "PosttrainPnPNovelFromTrayToPotSplitA"
    "PosttrainPnPNovelFromTrayToTieredbasketSplitA"
    "PosttrainPnPNovelFromTrayToTieredshelfSplitA"
)

case $SET_ID in
    1) TASK_NAMES=("${TASK_NAMES_SET_1[@]}") ;;
    2) TASK_NAMES=("${TASK_NAMES_SET_2[@]}") ;;
    3) TASK_NAMES=("${TASK_NAMES_SET_3[@]}") ;;
    4) TASK_NAMES=("${TASK_NAMES_SET_4[@]}") ;;
    *)
        echo "Invalid SET_ID: $SET_ID"
        echo "Usage: bash evaluation_checkpoint.sh [SET_ID]"
        exit 1
        ;;
esac

echo "Running evaluation for TASK SET $SET_ID: ${TASK_NAMES[*]}"
BASE_PATH="/home/changyeon/evaluations/${CKPT_ID}"
NUM_EPISODES=20
for TASK_NAME in "${TASK_NAMES[@]}"; do
    python3 scripts/simulation_service.py --client \
        --max_episode_steps 720 \
        --n_envs 1 \
        --n_episodes ${NUM_EPISODES} \
        --env_name gr1_unified/${TASK_NAME}_GR1ArmsAndWaistFourierHands_Env \
        --output_dir $BASE_PATH/stats/gr1_unified.${TASK_NAME}_GR1ArmsAndWaistFourierHands_Env \
        --video_dir $BASE_PATH/videos/gr1_unified.${TASK_NAME}_GR1ArmsAndWaistFourierHands_Env
done