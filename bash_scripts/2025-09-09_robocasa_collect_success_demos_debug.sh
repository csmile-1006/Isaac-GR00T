ACTION_HORIZON=16
TASK_NAME=$1
NUM_ENVS=$2
NUM_ROLLOUTS=$3
SERVER=${4:-"alin_slurm"}
NUM_PROCS=${5:-4}

if [ "$SERVER" == "rlwrld" ]; then
    ROOT_PATH=/virtual_lab/sjw_alinlab/changyeon/
else
    ROOT_PATH=/home/changyeon/
fi

source ${ROOT_PATH}/miniconda3/bin/activate gr00t
cd ${ROOT_PATH}/workspace/Isaac-GR00T

BASE_PATH=${ROOT_PATH}/debug/gr00tn15_robocasa/rollout_demos/
CKPT_PATH=${ROOT_PATH}/ckpts/gr00tn15_rbcs_bs32_60k

python scripts/collect_demo_robocasa.py \
    --host localhost \
    --port 5555 \
    --data_config single_panda_gripper \
    --action_horizon ${ACTION_HORIZON} \
    --embodiment_tag new_embodiment \
    --model_path ${CKPT_PATH} \
    --env_name ${TASK_NAME} \
    --num_episodes ${NUM_ROLLOUTS} \
    --collect_data \
    --reward_shaping \
    --noise 0.0 \
    --data_collection_path ${BASE_PATH}/${TASK_NAME}/data \
    --n_envs ${NUM_ENVS}

cd ${ROOT_PATH}/workspace/robocasa
OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python robocasa/scripts/dataset_states_to_obs_robust.py \
    --dataset ${BASE_PATH}/${TASK_NAME}/data/demo.hdf5 \
    --camera_width 256 \
    --camera_height 256 \
    --generative_textures \
    --randomize_cameras \
    --shaped --copy_rewards --copy_dones \
    --num_procs ${NUM_PROCS}

cd ${ROOT_PATH}/workspace/Isaac-GR00T
python scripts/convert_hdf5_to_lerobot.py \
    --src_hdf5_path ${BASE_PATH}/${TASK_NAME}/data/demo_gentex_im256_randcams.hdf5 \
    --output_path ${BASE_PATH}/${TASK_NAME}/lerobot \
    --task_name ${TASK_NAME} \
    --chunks_size 1000 \
    --num_episodes ${NUM_ROLLOUTS} \
    --num_video_workers ${NUM_PROCS}
