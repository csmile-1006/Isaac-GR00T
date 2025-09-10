TASK_NAME=$1
NUM_EPISODES=$2

CMD="python gr00t/data/data_merger.py merge \
  --datasets '/home/changyeon/data/robocasa_full/single_panda_gripper.${TASK_NAME}/' \
  --output_dir /home/changyeon/data/robocasa_dataset/${TASK_NAME}_num${NUM_EPISODES} \
  --num_episodes ${NUM_EPISODES} \
  --verbose"

echo $CMD
eval $CMD