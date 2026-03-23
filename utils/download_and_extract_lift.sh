#!/bin/bash
# Downloads lift proficient-human raw dataset and extracts image observations from it.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
ROBOMIMIC_SCRIPTS="$VENV/src/robomimic/robomimic/scripts"
DATASET_DIR="$SCRIPT_DIR/dataset/lift/ph"
RAW_HDF5="$DATASET_DIR/demo_v15.hdf5"
IMAGE_HDF5="$DATASET_DIR/image_v15.hdf5"

echo "=== Step 1: Download lift_ph raw dataset ==="
"$VENV/bin/python" "$ROBOMIMIC_SCRIPTS/download_datasets.py" \
    --tasks lift \
    --dataset_types ph \
    --hdf5_types raw \
    --download_dir "$SCRIPT_DIR/dataset"

echo ""
echo "=== Step 2: Extract 84x84 image observations ==="
"$VENV/bin/python" "$ROBOMIMIC_SCRIPTS/dataset_states_to_obs.py" \
    --dataset "$RAW_HDF5" \
    --output_name image.hdf5 \
    --done_mode 2 \
    --camera_names agentview robot0_eye_in_hand \
    --camera_height 84 \
    --camera_width 84 \
    --compress \
    --exclude-next-obs

echo ""
echo "Done! Image dataset saved to: $IMAGE_HDF5"
