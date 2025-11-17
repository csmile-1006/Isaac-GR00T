TASK_NAMES=(
    "CoffeeSetupMug"
    "PnPMicrowaveToCounter"
    "TurnOffStove"
    "PnPCounterToMicrowave"
)

DEMO_DATASETS=""
for TASK_NAME in "${TASK_NAMES[@]}"; do
    DEMO_DATASETS+="/home/changyeon/data/deas_robocasa/demos/${TASK_NAME}/ "
done

ROLLOUT_DATASETS=""
for TASK_NAME in "${TASK_NAMES[@]}"; do
    ROLLOUT_DATASETS+="/home/changyeon/data/deas_robocasa/rollouts/${TASK_NAME}/ "
done

SUCCESS_DATASETS=""
for TASK_NAME in "${TASK_NAMES[@]}"; do
    SUCCESS_DATASETS+="/home/changyeon/data/deas_robocasa/success_rollouts/${TASK_NAME}/ "
done

CMD="python gr00t/data/data_merger.py merge \
  --datasets '${DEMO_DATASETS} \
            ${ROLLOUT_DATASETS} \
             ' \
  --output_dir ~/data/changyeon/deas_robocasa_demo_rollouts \
  --verbose"

echo $CMD
eval $CMD