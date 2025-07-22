TASK_NAME=$1
NUM_ROLLOUTS=$2

python scripts/convert_hdf5_to_lerobot.py \
    --src_hdf5_path ./evaluation/${TASK_NAME}_singletask_fullft_as16/data/demo_gentex_im256_randcams.hdf5 \
    --output_path /home/changyeon/robocasa_dataset/rollouts/${TASK_NAME}_num${NUM_ROLLOUTS} \
    --task_name $TASK_NAME \
    --chunks_size 300 \
    --meta_path /home/changyeon/robocasa_dataset/${TASK_NAME}_num300/meta/