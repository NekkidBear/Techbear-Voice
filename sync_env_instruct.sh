#!/usr/bin/env bash
# .env can't source or reference another file -- python-dotenv reads it as
# static KEY=VALUE text, no shell evaluation, no includes. So base_voice.sh
# and .env's INSTRUCT= can't literally point at each other. This script is
# the practical substitute: base_voice.sh stays the one place you actually
# edit the voice description, and this re-syncs .env's INSTRUCT to match
# with one command instead of hand copy-pasting a long string between two
# differently-formatted files.
#
# Only worth running if you use run_markdown_tts.py directly, without
# --instruct, outside of run_test_scenes.sh / run_full_monologue.sh --
# those two always pass --instruct explicitly and never fall back to
# .env's INSTRUCT, so this doesn't affect them either way.
#
# Usage: ./sync_env_instruct.sh [path-to-.env]   (defaults to ./.env)

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/base_voice.sh"

ENV_FILE="${1:-.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "No ${ENV_FILE} found. Nothing to sync." >&2
  exit 1
fi

if grep -q '^INSTRUCT=' "${ENV_FILE}"; then
  awk -v val="${BASE_VOICE}" '
    BEGIN { done = 0 }
    /^INSTRUCT=/ { print "INSTRUCT=" val; done = 1; next }
    { print }
    END { if (!done) print "INSTRUCT=" val }
  ' "${ENV_FILE}" > "${ENV_FILE}.tmp" && mv "${ENV_FILE}.tmp" "${ENV_FILE}"
  echo "Updated INSTRUCT in ${ENV_FILE} from base_voice.sh"
else
  echo "INSTRUCT=${BASE_VOICE}" >> "${ENV_FILE}"
  echo "Appended INSTRUCT to ${ENV_FILE} from base_voice.sh"
fi