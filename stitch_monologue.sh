#!/usr/bin/env bash
# Stitches the 29 rendered scene wavs (output/monologue/sceneNN.wav) into
# one continuous take: output/monologue/full_take.wav
#
# Requires ffmpeg (brew install ffmpeg on macOS).
# Run after run_full_monologue.sh has produced all 29 files.

set -euo pipefail

OUT_DIR="output/monologue"
LIST_FILE="${OUT_DIR}/concat_list.txt"

: > "${LIST_FILE}"
for f in "${OUT_DIR}"/scene*.wav; do
  echo "file '$(basename "${f}")'" >> "${LIST_FILE}"
done

ffmpeg -y -f concat -safe 0 -i "${LIST_FILE}" -c copy "${OUT_DIR}/full_take.wav"

echo "Stitched take saved to ${OUT_DIR}/full_take.wav"
