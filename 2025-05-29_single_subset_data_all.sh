NUM_EPISODES=$1

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
  CMD="python gr00t/data/data_merger.py merge \
    --datasets '/mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.${TASK_NAME}/' \
    --output_dir ~/robocasa_dataset/${TASK_NAME}_num${NUM_EPISODES} \
    --num_episodes ${NUM_EPISODES} \
    --verbose"

  echo $CMD
  eval $CMD
done