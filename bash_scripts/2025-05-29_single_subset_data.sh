TASK_NAME=$1
NUM_EPISODES=$2

CMD="python gr00t/data/data_merger.py merge \
  --datasets '/mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.${TASK_NAME}/' \
  --output_dir ~/robocasa_dataset/${TASK_NAME}_num${NUM_EPISODES} \
  --num_episodes ${NUM_EPISODES} \
  --verbose"

echo $CMD
eval $CMD