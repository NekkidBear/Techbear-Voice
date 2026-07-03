# TechBear Voice Kit

Two related pieces for `Techbear-Voice`, built around one constraint in
`run_markdown_tts.py`: `extract_stage_directions()` pulls every `_(...)_`
direction out of whatever file it's given and joins them all into one
instruct string, with no regard for where in the document each one sits.
Point it at a file with more than one register in it and every beat gets
the same flattened instruct. The fix in both pieces below is the same:
one file per beat, so each call only ever has one direction to extract.

## 1. Dial-in test kit (`docs/test_scenes/`)

Six short, deliberately mismatched-register clips pulled from the social
intro, for fast `--instruct` iteration without rendering the whole thing:
dry regal, gently panicked, warm growl, conspiratorial whisper, quick
snap, confident warm.

These files carry **no** embedded stage direction — the delivery note for
each is hand-written into `run_test_scenes.sh` and passed via
`--instruct`, appended to a constant `BASE_VOICE` string that stays fixed
across all six. Good for quickly A/B-ing how a specific delivery
descriptor lands before committing to a full voice-design prompt.

Run: `./run_test_scenes.sh` → `output/voice_test/sceneNN_*.wav`

## 2. Full monologue, scene-split (`docs/monologue_scenes/`)

The complete social intro (from `Techbear_Social_Intro.docx`), split into
29 scenes at each stage-direction boundary. Unlike the test kit, each of
these files **does** carry its one embedded `_(direction)_` line — since
each file only has one direction, the tool's own auto-extraction handles
the merge correctly without needing a hand-written note per beat.
`run_full_monologue.sh` passes the same `BASE_VOICE` string to all 29
calls; `stitch_monologue.sh` (needs `ffmpeg`) concatenates the renders
into one continuous take afterward.

Run: `./run_full_monologue.sh` → `output/monologue/sceneNN.wav`, then
optionally `./stitch_monologue.sh` → `output/monologue/full_take.wav`

`scenes_manifest.json` lists all 29 scenes with their direction and
output filename, for anything you want to script against later.

## Keep `BASE_VOICE` in sync

Both run scripts define their own `BASE_VOICE` string independently.
Once you've dialed it in against the six test scenes, copy the final
version into `run_full_monologue.sh` too before doing a full take —
they're not linked, so an edit in one won't propagate to the other.

## If a clip comes back as garbled noise (or garbles briefly, then recovers)

Confirmed in practice, not just in theory: re-running the exact same
scene file, same instruct, same everything produced a clean result where
it had been garbled, and moved the garbling to a *different* scene on
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
re-render just that one directly:

```bash
python run_markdown_tts.py \
  --input docs/monologue_scenes/sceneNN_slug.md \
  --output output/monologue/sceneNN.wav \
  --instruct "$BASE_VOICE" \
  --max-new-tokens 1000
```

(`$BASE_VOICE` is the same string defined at the top of
`run_full_monologue.sh` — copy it in, or `source` the script's variable
section if you'd rather not retype it.) Repeat until that scene lands
clean, same as any other take.

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
`docs/` folder, and the three `.sh` files plus `scenes_manifest.json` into
the repo root. Activate `.venv`, confirm `.env` is set up per the repo's
own README, `chmod +x *.sh`, then run either script.
