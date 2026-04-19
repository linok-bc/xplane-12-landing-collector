#!/bin/bash
DONE_FLAG="/home/linok/Documents/xp12_dataset/DONE"
XPLANE="/home/linok/.local/share/Steam/steamapps/common/X-Plane 12/X-Plane-x86_64"

# installed via Steam; hacky way to avoid. Steam should be running in the background
export SteamAppId=2014780
export SteamGameId=2014780

# force NVIDIA GPU
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

while [ ! -f "$DONE_FLAG" ]; do
    start=$(date +%s)
    "$XPLANE" --window=1280x720
    elapsed=$(($(date +%s) - start))
    [ $elapsed -lt 30 ] && sleep 30
    sleep 5
done
echo "Dataset complete."
