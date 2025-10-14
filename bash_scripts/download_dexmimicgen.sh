#!/bin/bash
set -e
TARGET_DIR=$1

# === 설정 ===
REPO_URL="https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"

# === 필요한 폴더 목록 ===
FOLDERS=(
  "bimanual_panda_gripper.Threading"
  "bimanual_panda_hand.LiftTray"
  "bimanual_panda_gripper.ThreePieceAssembly"
  "bimanual_panda_gripper.Transport"
  "bimanual_panda_hand.BoxCleanup"
  "bimanual_panda_hand.DrawerCleanup"
  "gr1_arms_only.CanSort"
  "gr1_full_upper_body.Coffee"
  "gr1_full_upper_body.Pouring"
)

# === clone & sparse checkout ===
echo "[1/4] Cloning repo with sparse checkout enabled..."
git clone --filter=blob:none --sparse "$REPO_URL" "$TARGET_DIR"

cd "$TARGET_DIR"

echo "[2/4] Setting up sparse-checkout..."
git sparse-checkout init --cone
git sparse-checkout set "${FOLDERS[@]}"

echo "[3/4] Pulling selected files..."
git lfs pull --include "${FOLDERS[*]}"

echo "[4/4] Done!"
echo "Downloaded folders are inside: $(pwd)"
