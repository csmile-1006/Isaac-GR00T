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

BASE_PATH="/home/changyeon/data/gr1_data"
NUM_EPISODES=$1

for TASK_NAME in "${TASK_NAMES[@]}"; do
    echo "Processing $TASK_NAME"
    CMD="python gr00t/data/data_merger.py merge \
        --datasets $BASE_PATH/num1000/LeRobot/gr1_unified.$TASK_NAME \
        --output_dir $BASE_PATH/num${NUM_EPISODES}/LeRobot/gr1_unified.$TASK_NAME \
        --num_episodes ${NUM_EPISODES} \
        --verbose"
    echo $CMD
    eval $CMD
done