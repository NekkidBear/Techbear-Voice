#!/usr/bin/env bash
# Renders the full TechBear social intro monologue as 29 separate scene
# clips, one per stage-direction beat, then (optionally) stitches them
# into a single wav.
#
# WHY PER-SCENE: run_markdown_tts.py's extract_stage_directions() pulls
# every _(...)_ direction out of whatever file you point it at and mashes
# them all into one instruct string, regardless of where in the document
# they appear. Point it at the whole monologue in one call and TechBear's
# "dry regal" open and his "angry rapid-fire" dean rant both get the same
# flattened instruct. Splitting into one file per beat means each call's
# single embedded direction is the only one the script has to extract —
# so its own auto-merge logic (instruct + "Use the following stage
# directions to guide tone: ...") does exactly what you want it to,
# scene by scene, without needing to hand-write a delivery note per beat.
#
# Run from the repo root, with .venv activated:
#   chmod +x run_full_monologue.sh
#   ./run_full_monologue.sh
#
# Copy docs/monologue_scenes/*.md into your docs/ folder before running.

set -euo pipefail

# Base voice identity, held constant across all 29 calls — same as
# run_test_scenes.sh. Each call's --instruct is this string; the script
# appends that scene's one embedded stage direction automatically.
BASE_VOICE="Middle-aged male voice, mid-American accent with a slight Southern warmth. Resonant and slightly gravelly, with mild vocal fry. Nasal, theatrical delivery — campy and sly, but grounded by genuine technical authority and warmth. Measured speaking pace. Sarcastic edge with a kind undertone. More dry wit than exuberance."

# Caps runaway generation. generate_custom_voice() defaults to
# max_new_tokens=2048 when unset, and sampling (do_sample=True,
# temperature=0.9) occasionally fails to hit a stop token, producing
# long garbled audio instead of a short clean clip -- this is what
# happened to scenes 1 and 6 of the dial-in test kit. Monologue scenes
# run longer than the test kit (up to ~14 lines), so the cap here is
# higher. Raise it if a clip sounds cut off, lower it if a clip comes
# back garbled and oversized.
MAX_NEW_TOKENS=1000

SCENE_DIR="docs/monologue_scenes"
OUT_DIR="output/monologue"
mkdir -p "${OUT_DIR}"

for scene_path in "${SCENE_DIR}"/scene*.md; do
  scene_file="$(basename "${scene_path}")"
  # scene01_dry-slow-regal-like.md -> scene01.wav
  scene_num="$(echo "${scene_file}" | grep -oE '^scene[0-9]+')"
  out_file="${scene_num}.wav"
  echo "=== Rendering ${scene_file} -> ${out_file} ==="
  python run_markdown_tts.py \
    --input "${SCENE_DIR}/${scene_file}" \
    --output "${OUT_DIR}/${out_file}" \
    --instruct "${BASE_VOICE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}"
done

echo ""
echo "All 29 scenes rendered to ${OUT_DIR}/"
echo "To stitch into one continuous take (requires ffmpeg):"
echo "  ./stitch_monologue.sh"
