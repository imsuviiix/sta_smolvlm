#!/bin/bash
# Usage: run_sam2_vlm.sh <video_path> [output_dir]
VIDEO=$1
if [ -z "$VIDEO" ]; then
  echo "Usage: $0 <video_path> [output_dir]" >&2
  exit 1
fi
OUT_DIR=${2:-sam2_results}
python sam2_vlm_pipeline.py "$VIDEO" --output_dir "$OUT_DIR" "${@:3}"
