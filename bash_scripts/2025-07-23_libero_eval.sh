#!/bin/bash
#SBATCH --job-name=eval_libero
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=background
#SBATCH --array=0-9
#SBATCH --output=out/%j-eval_libero.out
#SBATCH --error=out/%j-eval_libero.err


TASK_NAME=$1
CKPT_STEP=$2

source /home/changyeon/miniconda3/bin/activate libero
cd /home/changyeon/workspace/Isaac-GR00T

BASE_DIR=/home/changyeon/workspace

CKPT_DIR="~/gr00t_n1_5_ckpt/libero/$TASK_NAME_$SLURM_ARRAY_TASK_ID/checkpoint-$CKPT_STEP"

CKPT_PATH="$CKPT_DIR"

echo "[i] Evaluating: CKPT_STEP=$CKPT_STEP..."

python scripts/inference_service.py \
    --port=822$SLURM_ARRAY_TASK_ID \
    --model_path=$CKPT_PATH \
    --data_config=libero \
    --server \
    &
SERVE_PID=$!


TASK_NAMES=("libero_10" "libero_goal" "libero_object" "libero_spatial")
MAIN_PIDS=()

sleep 10

for TASK_NAME in "${TASK_NAMES[@]}"; do
    OUTPUT_DIR="$BASE_DIR/output/libero/$TASK_NAME/gr00t_n1_5_libero-debug/$CKPT_STEP"
    mkdir -p "$OUTPUT_DIR"
    python gr00t/eval/libero/eval_taskwise_gr00t.py \
        --args.task-suite-name $TASK_NAME \
        --args.video-out-path $OUTPUT_DIR \
        --args.task_idx=$SLURM_ARRAY_TASK_ID \
        --args.port=822$SLURM_ARRAY_TASK_ID \
        >& "$OUTPUT_DIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done


# Wait on just the main.py processes
for pid in "${MAIN_PIDS[@]}"; do
    wait "$pid"
done

# Kill serve_policy once those tasks finish
kill "$SERVE_PID"
echo "[i] Finished CKPT_STEP=$CKPT_STEP on GPU $SLURM_ARRAY_TASK_ID."