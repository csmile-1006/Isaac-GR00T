NUM_EPISODES=$1

CMD="python gr00t/data/data_merger.py merge \
  --datasets '/mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.TurnSinkSpout/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.TurnOnStove/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.TurnOnSinkFaucet/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.TurnOnMicrowave/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.TurnOffStove/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.TurnOffSinkFaucet/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.TurnOffMicrowave/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.PnPStoveToCounter/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.PnPSinkToCounter/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.PnPMicrowaveToCounter/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.PnPCounterToStove/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.PnPCounterToSink/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.PnPCounterToMicrowave/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.PnPCounterToCab/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.PnPCabToCounter/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.OpenSingleDoor/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.OpenDrawer/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.OpenDoubleDoor/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.CoffeeSetupMug/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.CoffeeServeMug/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.CoffeePressButton/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.CloseSingleDoor/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.CloseDrawer/ \
             /mnt/nas/slurm_account/visitor_jw/gr00t_dataset/single_panda_gripper.CloseDoubleDoor/ \
             ' \
  --output_dir ~/robocasa_dataset/mix${NUM_EPISODES} \
  --num_episodes ${NUM_EPISODES} \
  --verbose"

echo $CMD
eval $CMD