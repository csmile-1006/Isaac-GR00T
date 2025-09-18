TASK_NAME=$1

ACTION_HORIZON=16
NUM_ROLLOUTS=$2
SERVER=${3:-"alin_slurm"}
NUM_PROCS=${4:-4}

if [ "$SERVER" == "rlwrld" ]; then
    ROOT_PATH=/virtual_lab/sjw_alinlab/changyeon/
else
    ROOT_PATH=/home/changyeon/
fi

source ${ROOT_PATH}/miniconda3/bin/activate gr00t
BASE_PATH=${ROOT_PATH}/gr00tn15_robocasa/rollouts/

cd ${ROOT_PATH}/workspace/robocasa
OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python robocasa/scripts/dataset_states_to_obs_robust.py \
    --dataset ${BASE_PATH}/${TASK_NAME}_num${NUM_ROLLOUTS}/data/demo.hdf5 \
    --camera_width 256 \
    --camera_height 256 \
    --generative_textures \
    --randomize_cameras \
    --shaped --copy_rewards --copy_dones \
    --num_procs ${NUM_PROCS}

cd ${ROOT_PATH}/workspace/Isaac-GR00T
python scripts/convert_hdf5_to_lerobot.py \
    --src_hdf5_path ${BASE_PATH}/${TASK_NAME}_num${NUM_ROLLOUTS}/data/demo_gentex_im256_randcams.hdf5 \
    --output_path ${BASE_PATH}/${TASK_NAME}_num${NUM_ROLLOUTS}/lerobot \
    --task_name ${TASK_NAME} \
    --chunks_size 1000 \
    --num_episodes ${NUM_ROLLOUTS} \
    --num_video_workers ${NUM_PROCS}
