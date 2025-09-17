TASK_NAMES=(
  "TurnSinkSpout"
  "TurnOnStove"
  "TurnOnSinkFaucet"
  "TurnOnMicrowave"
  "TurnOffStove"
  "TurnOffSinkFaucet"
  "TurnOffMicrowave"
  "PnPStoveToCounter"
  "PnPSinkToCounter"
  "PnPMicrowaveToCounter"
  "PnPCounterToStove"
  "PnPCounterToSink"
  "PnPCounterToMicrowave"
  "PnPCounterToCab"
  "PnPCabToCounter"
  "OpenSingleDoor"
  "OpenDrawer"
  "OpenDoubleDoor"
  "CoffeeSetupMug"
  "CoffeeServeMug"
  "CoffeePressButton"
  "CloseSingleDoor"
  "CloseDrawer"
  "CloseDoubleDoor"
)

for TASK_NAME in ${TASK_NAMES[@]}; do
    python gr00t/data/data_merger.py success_demos \
        --dataset_dir /home/changyeon/gr00tn15_robocasa/rollouts_num300/${TASK_NAME}/lerobot/ \
        --env_name ${TASK_NAME} \
        --output_dir /home/changyeon/gr00tn15_robocasa/rollout_demos_num300/${TASK_NAME}/lerobot/
done