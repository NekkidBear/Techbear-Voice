#!/usr/bin/env bash
# Guard against `source ./run_test_scenes.sh`: sourcing runs this in your
# current shell process rather than a subshell, so any failure below
# (combined with `set -e`) would exit YOUR terminal session, not just this
# script. Run it as `./run_test_scenes.sh` instead.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "Don't source this script -- run it instead: ./run_test_scenes.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# TechBear voice dial-in test — renders 6 short scenes, one per vocal register.
# Loads the model ONCE via render_batch.py and reuses it for all 12 renders
# (6 scenes x 2 takes), instead of shelling out to run_markdown_tts.py per
# clip and reloading the whole model from scratch each time. Repeated
# multi-GB model loads were the likely source of memory-pressure spikes
# in Activity Monitor.
#
# Run from the repo root, with .venv activated:
#   chmod +x run_test_scenes.sh
#   ./run_test_scenes.sh
#
# Drop docs/test_scenes/*.md, test_scenes_manifest.json, and
# render_batch.py into your repo root/docs before running.

set -euo pipefail

# Preflight: fail with a clear message instead of a bare "command not
# found" (exit 127) if the venv isn't active. `python` only exists once
# a venv is activated -- macOS doesn't ship a bare `python` by default,
# only `python3`.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "No virtualenv active. Activate it first, e.g.:" >&2
  echo "  source backend/venv/bin/activate" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on PATH even with a venv active. Check your venv." >&2
  exit 1
fi

# Base voice identity, sourced from .env's INSTRUCT rather than a
# separate base_voice.sh copy -- the two had already drifted out of
# sync once when they were kept in separate files.
set -a
source "$(dirname "${BASH_SOURCE[0]}")/.env"
set +a

# Caps runaway generation. generate_custom_voice() defaults to
# max_new_tokens=2048 when unset, and sampling (do_sample=True,
# temperature=0.9) occasionally fails to hit a stop token, producing
# long garbled audio instead of a short clean clip. These scenes are
# 3-5 short lines, so 400 is generous headroom -- raise it if a clip
# sounds cut off, lower it if you're still getting gobbledygook.
MAX_NEW_TOKENS=400

# Sampling is stochastic and occasionally garbles a clip -- confirmed to
# be pure per-call luck, not tied to any particular scene's text. Render
# two takes of each scene and keep whichever is clean.
TAKES=2

python3 render_batch.py \
  --scene-dir docs/test_scenes \
  --out-dir output/voice_test \
  --manifest test_scenes_manifest.json \
  --instruct "${INSTRUCT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --takes "${TAKES}"