#!/bin/bash

TASK_NAMES=(
    "PnPBottleToCabinetClose"
    "PnPCanToDrawerClose"
    "PnPCupToDrawerClose"
    "PnPMilkToMicrowaveClose"
    "PnPPotatoToMicrowaveClose"
    "PnPWineToCabinetClose"
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA"
    "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA"
    "PosttrainPnPNovelFromCuttingboardToPanSplitA"
    "PosttrainPnPNovelFromCuttingboardToPotSplitA"
    "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA"
    "PosttrainPnPNovelFromPlacematToBasketSplitA"
    "PosttrainPnPNovelFromPlacematToBowlSplitA"
    "PosttrainPnPNovelFromPlacematToPlateSplitA"
    "PosttrainPnPNovelFromPlacematToTieredshelfSplitA"
    "PosttrainPnPNovelFromPlateToBowlSplitA"
    "PosttrainPnPNovelFromPlateToCardboardboxSplitA"
    "PosttrainPnPNovelFromPlateToPanSplitA"
    "PosttrainPnPNovelFromPlateToPlateSplitA"
    "PosttrainPnPNovelFromTrayToCardboardboxSplitA"
    "PosttrainPnPNovelFromTrayToPlateSplitA"
    "PosttrainPnPNovelFromTrayToPotSplitA"
    "PosttrainPnPNovelFromTrayToTieredbasketSplitA"
    "PosttrainPnPNovelFromTrayToTieredshelfSplitA"
)

NUM_EPISODES=$1
ALL_DATASET_PATHS=()
BASE_PATH="/home/changyeon/data/gr1_data/num${NUM_EPISODES}/LeRobot"
for TASK_NAME in "${TASK_NAMES[@]}"; do
    ALL_DATASET_PATHS+=("$BASE_PATH/gr1_unified.$TASK_NAME")
done
CKPT_PATH="/home/changyeon/ckpts/gr1_num${NUM_EPISODES}"


python scripts/gr00t_finetune.py \
  --dataset-path "${ALL_DATASET_PATHS[@]}" \
  --num-gpus 2 --batch-size 120 --learning_rate 3e-5 \
  --output-dir $CKPT_PATH \
  --data-config fourier_gr1_arms_waist --embodiment_tag gr1 \
  --max-steps 30000 --save-steps 10000