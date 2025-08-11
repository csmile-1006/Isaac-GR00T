#!/bin/bash
#SBATCH --job-name=eval_libero
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=48G
#SBATCH --time=72:00:00
#SBATCH --output=/home/changyeon/slurm-logs/eval_libero/%j.out
#SBATCH --error=/home/changyeon/slurm-logs/eval_libero/%j.err


TASK_NAME=$1
CKPT_STEP=$2
TASK_INDEX=$3

echo "TASK_NAME: $TASK_NAME"
echo "CKPT_STEP: $CKPT_STEP"
echo "TASK_INDEX: $TASK_INDEX"

source /home/changyeon/miniconda3/bin/activate libero
cd /home/changyeon/workspace/Isaac-GR00T

BASE_DIR=/home/changyeon/workspace

CKPT_DIR="/home/changyeon/gr00t_n1_5_ckpt/libero/$TASK_NAME/step60000/checkpoint-$CKPT_STEP/"

CKPT_PATH="$CKPT_DIR"

echo "[i] Evaluating: CKPT_STEP=$CKPT_STEP..."

python scripts/inference_service.py \
    --port=8220 \
    --model_path=$CKPT_PATH \
    --data_config=libero \
    --embodiment_tag=new_embodiment \
    --server \
    &
SERVE_PID=$!


TASK_NAMES=("libero_10" "libero_goal" "libero_object" "libero_spatial")
MAIN_PIDS=()

sleep 10

OUTPUT_DIR="$BASE_DIR/output/libero/$TASK_NAME/gr00t_n1_5_libero-debug/$CKPT_STEP"
mkdir -p "$OUTPUT_DIR"
python scripts/eval_policy_libero.py \
    --args.task-suite-name $TASK_NAME \
    --args.video-out-path $OUTPUT_DIR \
    --args.task_idx=$TASK_INDEX \
    --args.port=8220 \
    >& "$OUTPUT_DIR/eval-$TASK_INDEX.log" &
MAIN_PIDS+=($!)


# Wait on just the main.py processes
for pid in "${MAIN_PIDS[@]}"; do
    wait "$pid"
done

# Kill serve_policy once those tasks finish
kill "$SERVE_PID"
echo "[i] Finished CKPT_STEP=$CKPT_STEP on GPU $SLURM_ARRAY_TASK_ID."