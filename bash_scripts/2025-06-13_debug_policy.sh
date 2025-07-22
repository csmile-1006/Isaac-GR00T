#!/bin/bash                                                                                                                                                  

NUM_DEMOS=$1
TASK_NAME=$2
ACTION_HORIZON=$3

python scripts/eval_policy_debug.py \
    --host localhost \
    --port 5555 \
    --data_config single_panda_gripper_rl \
    --action_horizon ${ACTION_HORIZON} \
    --video_backend decord \
    --dataset_path ~/robocasa_dataset/${TASK_NAME}_num${NUM_DEMOS}/ \
    --embodiment_tag new_embodiment \
