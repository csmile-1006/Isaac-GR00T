TASK_NAME=$1
NUM_EPISODES=$2
SERVER=$3

if [ "$SERVER" == "h100" ]; then
    DATA_PATH=/home/changyeon/data/robocasa_full/single_panda_gripper.${TASK_NAME}/
elif [ "$SERVER" == "alin_slurm" ]; then
    DATA_PATH=/home/changyeon/data/robocasa_dataset/${TASK_NAME}_num300
fi

CMD="python gr00t/data/data_merger.py merge \
  --datasets ${DATA_PATH} \
  --output_dir /home/changyeon/data/robocasa_dataset/${TASK_NAME}_num${NUM_EPISODES} \
  --num_episodes ${NUM_EPISODES} \
  --verbose"

echo $CMD
eval $CMD