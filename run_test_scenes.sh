#!/usr/bin/env bash
# TechBear voice dial-in test — renders 6 short scenes, one per vocal register,
# each with its own --instruct override so the base voice identity stays locked
# while delivery shifts per stage direction.
#
# Run from the repo root, with .venv activated:
#   chmod +x run_test_scenes.sh
#   ./run_test_scenes.sh
#
# Drop the docs/test_scenes/*.md files into your docs/ folder before running.

set -euo pipefail

# Base voice identity — pulled from qwen-config-example.txt. Kept constant
# across every scene call; only the delivery_note changes per scene.
BASE_VOICE="Middle-aged male voice, mid-American accent with a slight Southern warmth. Resonant and slightly gravelly, with mild vocal fry. Nasal, theatrical delivery — campy and sly, but grounded by genuine technical authority and warmth. Measured speaking pace. Sarcastic edge with a kind undertone. More dry wit than exuberance."

# Caps runaway generation. generate_custom_voice() defaults to
# max_new_tokens=2048 when unset, and sampling (do_sample=True,
# temperature=0.9) occasionally fails to hit a stop token, producing
# long garbled audio instead of a short clean clip. These scenes are
# 3-5 short lines, so 400 is generous headroom -- raise it if a clip
# sounds cut off, lower it if you're still getting gobbledygook.
MAX_NEW_TOKENS=400

# Sampling is stochastic and occasionally garbles a clip -- confirmed to
# be pure per-call luck, not tied to any particular scene's text (it hit
# scene 1 on one run, scene 2 on the next, with zero changes in between).
# Capping max-new-tokens catches full runaway generation, but a "garbled
# start that recovers into clean speech" clip isn't too long, just bad --
# no token cap fixes that. Cheapest reliable fix: render each scene twice
# and keep whichever take is clean.
TAKES=2

mkdir -p output/voice_test

run_scene () {
  local scene_file="$1"
  local out_stem="$2"
  local delivery_note="$3"
  for take in $(seq 1 "${TAKES}"); do
    echo "=== Rendering ${scene_file} (take ${take}) ==="
    python run_markdown_tts.py \
      --input "docs/test_scenes/${scene_file}" \
      --output "output/voice_test/${out_stem}_take${take}.wav" \
      --instruct "${BASE_VOICE} ${delivery_note}" \
      --max-new-tokens "${MAX_NEW_TOKENS}"
  done
}

run_scene "scene01_dry_regal.md" "scene01_dry_regal" \
  "For this line, deliver dry, slow, and regal — like Bea Arthur about to deliver a roast. Deadpan pacing, long pauses between phrases."

run_scene "scene02_gently_panicked.md" "scene02_gently_panicked" \
  "For this line, shift to gently flustered and anxious, C-3PO-style energy creeping in — faster cadence, slightly higher pitch tension."

run_scene "scene03_warm_growl.md" "scene03_warm_growl" \
  "For this line, drop into a warm, gravelly growl, Harvey Fierstein-style — affectionate and knowing, slower and huskier than the base voice."

run_scene "scene04_conspiratorial_whisper.md" "scene04_conspiratorial_whisper" \
  "For this line, deliver as a hushed, conspiratorial whisper — leaning in close, playful secrecy, slightly breathy."

run_scene "scene05_quick_snap.md" "scene05_quick_snap" \
  "For this line, deliver quick, snappy, and arch — Paul Lynde zing energy, rapid-fire with a sudden punchline lift at the end."

run_scene "scene06_confident_warm.md" "scene06_confident_warm" \
  "For this line, deliver warm, confident, and reassuring — steady pace, embracing tone, settling the audience."

echo ""
echo "All scenes rendered to output/voice_test/"
