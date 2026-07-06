#!/usr/bin/env bash
# Guard against `source ./run_full_monologue.sh`: sourcing runs this in
# your current shell process rather than a subshell, so any failure below
# (combined with `set -e`) would exit YOUR terminal session, not just this
# script. Run it as `./run_full_monologue.sh` instead.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "Don't source this script -- run it instead: ./run_full_monologue.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# Renders the full TechBear social intro monologue as 29 separate scene
# clips, one per stage-direction beat, then (optionally) stitches them
# into a single wav. Loads the model ONCE via render_batch.py and reuses
# it for all 29 renders, instead of shelling out to run_markdown_tts.py
# per scene and reloading the whole model from scratch 29 times.
#
# WHY PER-SCENE: run_markdown_tts.py's extract_stage_directions() pulls
# every _(...)_ direction out of whatever file you point it at and mashes
# them all into one instruct string, regardless of where in the document
# they appear. Point it at the whole monologue in one call and TechBear's
# "dry regal" open and his "angry rapid-fire" dean rant both get the same
# flattened instruct. Splitting into one file per beat means each scene's
# single embedded direction is the only one there is to extract — so the
# tool's own merge logic (instruct + "Use the following stage directions
# to guide tone: ...", reused here from run_markdown_tts.py) does exactly
# what you want, scene by scene, without hand-written notes per beat.
#
# MODE: set explicitly below rather than relying on .env's MODEL_ID, since
# .env's MODEL_ID is meant as a fallback default and gets left on whatever
# was last used for ad-hoc comparison testing. A 29-scene render is
# expensive enough that "which mode did this actually run in" should
# never be a guess.
#
# Run from the repo root, with .venv activated:
#   chmod +x run_full_monologue.sh
#   ./run_full_monologue.sh
#
# Copy docs/monologue_scenes/*.md, scenes_manifest.json, and
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

# Which render_batch.py mode this run uses: custom, design, or clone.
# Change this line (or override with MODE=design ./run_full_monologue.sh)
# rather than editing .env -- keeps the choice visible and intentional
# for a run this expensive.
MODE="${MODE:-clone}"

# Caps runaway generation. generate_custom_voice() defaults to
# max_new_tokens=2048 when unset, and sampling (do_sample=True,
# temperature=0.9) occasionally fails to hit a stop token, producing
# long garbled audio instead of a short clean clip. Monologue scenes run
# longer than the test kit (up to ~14 lines), so the cap here is higher.
MAX_NEW_TOKENS=1000

# One take per scene by default -- doubling all 29 renders to catch what's
# usually a one-scene problem isn't worth it. If a specific scene comes
# back bad (or you want to spot-check before committing to the full 29),
# use --only <out_stem> to render just that one scene:
#   python3 render_batch.py --scene-dir docs/monologue_scenes \
#     --out-dir output/monologue --manifest scenes_manifest.json \
#     --mode "$MODE" --instruct "$INSTRUCT" --max-new-tokens 1000 --takes 1 \
#     --only scene01
# (render_batch.py still loads the model once for that single call --
# fine for a one-off retry, just not for the full 29-scene batch.)
TAKES=1

echo "Rendering full monologue in --mode ${MODE}"

python3 render_batch.py \
  --scene-dir docs/monologue_scenes \
  --out-dir output/monologue \
  --manifest scenes_manifest.json \
  --mode "${MODE}" \
  --instruct "${INSTRUCT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --takes "${TAKES}"

echo ""
echo "All 29 scenes rendered to output/monologue/ (mode: ${MODE})"
echo "To stitch into one continuous take (requires ffmpeg):"
echo "  ./stitch_monologue.sh"
