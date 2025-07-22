TASK_NAME=$1
NUM_EPISODES=$2
FAIL_EPISODES=$3

CMD="python gr00t/data/data_merger.py merge \
  --datasets '/home/changyeon/robocasa_dataset/${TASK_NAME}_num${NUM_EPISODES} \
             /home/changyeon/robocasa_dataset/rollouts/${TASK_NAME}_num${FAIL_EPISODES} \
             ' \
  --output_dir ~/robocasa_dataset/${TASK_NAME}_mixed_succ${NUM_EPISODES}_fail${FAIL_EPISODES} \
  --num_episodes ${NUM_EPISODES} \
  --verbose"

echo $CMD
eval $CMD