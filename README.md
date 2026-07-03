# TechBear Voice Kit

Two related pieces for `Techbear-Voice`, built around two constraints:

1. `run_markdown_tts.py`'s `extract_stage_directions()` pulls every
   `_(...)_` direction out of whatever file it's given and joins them
   all into one instruct string, with no regard for where in the
   document each one sits. Point it at a file with more than one
   register in it and every beat gets the same flattened instruct. Fix:
   one file per beat, so each call only ever has one direction to
   extract.
2. `run_markdown_tts.py` loads the model fresh on every invocation. Call
   it once per scene (as the original per-scene shell loop did) and
   you're loading multiple GB of weights onto MPS over and over — each
   load is a burst allocation, which is the likely source of brief
   memory-pressure spikes in Activity Monitor even when overall memory
   used looks fine. Fix: `render_batch.py` loads the model **once** and
   renders every scene (and every take) in that same process.

## `render_batch.py`

Loads `MODEL_ID` once, then iterates a directory of scene markdown
files, reusing `extract_stage_directions()` / `markdown_to_text()` /
`parse_dtype()` imported directly from `run_markdown_tts.py` so behavior
matches it exactly — this isn't a reimplementation, it's the same
per-scene logic the original script runs, just looped in-process instead
of relaunched per subprocess. Place it in the repo root, next to
`run_markdown_tts.py`, so the import resolves.

Takes a `--manifest` JSON file (a list of `{file, out_stem?,
instruct_suffix?}`) so scene-specific delivery notes — like the test
kit's hand-written ones — can ride along without hardcoding them into
the script. `scenes_manifest.json` (monologue) and
`test_scenes_manifest.json` (dial-in kit) are both included and already
wired up in the two run scripts below.

## 1. Dial-in test kit (`docs/test_scenes/`)

Six short, deliberately mismatched-register clips pulled from the social
intro, for fast `--instruct` iteration without rendering the whole thing:
dry regal, gently panicked, warm growl, conspiratorial whisper, quick
snap, confident warm.

These files carry **no** embedded stage direction — the delivery note
for each lives in `test_scenes_manifest.json` (`instruct_suffix`),
appended to a constant `BASE_VOICE` string that stays fixed across all
six. Good for quickly A/B-ing how a specific delivery descriptor lands
before committing to a full voice-design prompt.

Run: `./run_test_scenes.sh` → `output/voice_test/sceneNN_*_takeN.wav`

## 2. Full monologue, scene-split (`docs/monologue_scenes/`)

The complete social intro (from `Techbear_Social_Intro.docx`), split into
29 scenes at each stage-direction boundary. Unlike the test kit, each of
these files **does** carry its one embedded `_(direction)_` line — since
each file only has one direction, `extract_stage_directions()` handles
the merge correctly on its own without needing a hand-written note per
beat. `run_full_monologue.sh` passes the same `BASE_VOICE` string for all
29 scenes; `stitch_monologue.sh` (needs `ffmpeg`) concatenates the
renders into one continuous take afterward.

Run: `./run_full_monologue.sh` → `output/monologue/sceneNN.wav`, then
optionally `./stitch_monologue.sh` → `output/monologue/full_take.wav`

`scenes_manifest.json` lists all 29 scenes with their direction and
output filename — this is what `render_batch.py` reads to drive the run,
and it's also handy for anything else you want to script against later.

## `base_voice.sh`

Both run scripts now `source base_voice.sh` for `BASE_VOICE` instead of
each declaring their own copy — the two copies had already drifted out
of sync once, so there's exactly one definition now.

The current wording is a revision, not the original. The baseline pulled
from `qwen-config-example.txt` said "campy and sly" once and then closed
with "More dry wit than exuberance" — a line actively pushing the model
_away_ from the flamboyant diva energy `character_identity.md` and
`character_voice.md` both call for ("grand diva," "total queer
confidence," RuPaul-level showmanship, the "Diva Shift" register). If
renders start drifting back toward flat/reserved delivery again, that's
the file to check first — and the character docs are the source of
truth to check it against, not just ear.

`base_voice.sh` isn't wired into `.env`'s `INSTRUCT`, and can't be —
`.env` is read by `python-dotenv` as static `KEY=VALUE` text, no shell
evaluation or includes, so `INSTRUCT=./base_voice.sh` would just set
`INSTRUCT` to that literal string rather than the file's contents. This
only matters if you ever run `run_markdown_tts.py` directly without
`--instruct` (both run scripts here always pass `--instruct` explicitly
and never touch `.env`'s `INSTRUCT`). If you do want `.env` to match,
run `./sync_env_instruct.sh` — it rewrites `.env`'s `INSTRUCT=` line
from `base_voice.sh` in one command instead of hand copy-pasting.

## Brief red memory-pressure spikes in Activity Monitor

Likely explained by the repeated model loads described above, not by the
model simply being large for 16GB unified memory. Each `python
run_markdown_tts.py` subprocess call loads the full model onto MPS from
scratch; doing that 12 times (test kit) or 29 times (monologue) back to
back means 12 or 29 multi-GB burst allocations, which is a plausible
match for "brief spikes, then settles" even while steady-state usage
looks fine. `render_batch.py` loads once and reuses the same in-memory
model for every scene and take, which should smooth this out and run
noticeably faster besides. Watch for a different pattern this enables,
though: the old approach couldn't accumulate memory across scenes since
each process exited before the next started; a single long-lived process
theoretically could if the underlying generation loop doesn't fully
release intermediate tensors between calls. `render_batch.py` calls
`torch.mps.empty_cache()` after every render as insurance against that.
If you ever see memory climbing steadily over a long run rather than
spiking and settling, that's the pattern to flag as different from the
"ok" brief-spike behavior.

Two other things worth a look if spikes persist:

- `.env` has `DTYPE=float32`, roughly double the memory footprint of the
  repo's own recommended `float16` for Mac. Worth switching regardless.
- Malwarebytes' real-time scanner can cause its own transient spikes
  when it scans newly-written files as they land — check its own
  CPU/memory in Activity Monitor at the same timestamps as your red
  spikes, or exclude `~/.cache/huggingface` and your `output/` folder
  from real-time scanning.

## If a clip comes back as garbled noise (or garbles briefly, then recovers)

Confirmed in practice, not just in theory: re-running the exact same
scene file, same instruct, same everything produced a clean result where
it had been garbled, and moved the garbling to a _different_ scene on
the next run. This is `generate_custom_voice()`'s stochastic sampling
(`do_sample=True`, `temperature=0.9`, `top_p=1.0`) occasionally failing
to land cleanly — sometimes as a full runaway (keeps generating for the
whole `max_new_tokens` budget, which the token caps above address),
sometimes as a garbled few seconds at the start before the model finds
its footing and the rest plays correctly. The second kind isn't a length
problem, so `--max-new-tokens` alone won't fix it — it's genuinely
per-call luck, independent of scene content.

**Test kit (`run_test_scenes.sh`):** now renders two takes of each scene
by default (`scene0N_..._take1.wav`, `..._take2.wav`) — cheap to do
twice since these are six short clips. Listen to both, keep the clean
one. Bump `TAKES` at the top of the script if you want more insurance.

**Full monologue (`run_full_monologue.sh`):** left at one take per scene
by default — doubling all 29 renders every time is expensive for what's
usually a one- or two-scene problem. If a specific scene comes back bad,
re-render just that one directly (still goes through `render_batch.py`,
just with a one-entry scene dir or a trimmed manifest — simplest is to
copy the one bad scene file into its own temp folder):

```bash
mkdir -p /tmp/retry_scene && cp docs/monologue_scenes/sceneNN_slug.md /tmp/retry_scene/
python render_batch.py \
  --scene-dir /tmp/retry_scene \
  --out-dir output/monologue \
  --instruct "$BASE_VOICE" \
  --max-new-tokens 1000 \
  --takes 1
```

(`$BASE_VOICE` is the same string defined at the top of
`run_full_monologue.sh`.) The output filename will match the scene's
markdown filename stem rather than `sceneNN.wav` in this ad hoc form —
rename after, or restore in a manifest if you want the original name to
land automatically. Repeat until that scene lands clean.

## Two things worth knowing about the source script

1. **Two directions in `docs/example.md` don't use underscores** —
   `(Deadpan, then building into theatrical)` and `(Beat. Sighs, then
leans in with dry nostalgia)` are plain `(...)` rather than `_(...)_`.
   `extract_stage_directions()` only matches the underscored form, so if
   you ever run `docs/example.md` through the tool as-is, those two
   directions won't get pulled into the instruct — they'll leak into the
   spoken text and get read aloud as literal parenthetical text instead.
   Not an issue here; every scene file above uses consistent `_(...)_`
   formatting (see scenes 23 and 24 in `monologue_scenes/`).
2. **`(Shudder)`** (after "Poor Eddie.") is a direction with no dialogue
   of its own before the next direction takes over. Dropped rather than
   given its own wordless scene — it reads as a beat you'd play live or
   splice in as a non-verbal audio cue during editing, not something TTS
   needs to voice. Flag if you want it handled differently.

## Setup

Copy `docs/test_scenes/` and `docs/monologue_scenes/` into your repo's
`docs/` folder, and `render_batch.py`, `base_voice.sh`, both `.sh` run
scripts, and both `*_manifest.json` files into the repo root
(`render_batch.py` must sit next to `run_markdown_tts.py` for its import
to resolve). Activate `.venv`, confirm `.env` is set up per the repo's
own README, `chmod +x *.sh`, then run either script.
