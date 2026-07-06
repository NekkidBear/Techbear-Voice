#!/usr/bin/env python3
"""
Renders every scene in a directory with the model loaded ONCE, instead of
run_test_scenes.sh / run_full_monologue.sh's approach of shelling out to
`python run_markdown_tts.py` per scene (and per take), which reloads the
full model from disk/into MPS memory on every single call. Each of those
loads is a multi-GB burst allocation -- almost certainly what's showing
up as brief memory-pressure spikes in Activity Monitor. Loading once and
looping in-process should smooth that out and be noticeably faster.

Reuses markdown_to_text / extract_stage_directions / parse_dtype directly
from run_markdown_tts.py rather than reimplementing them, so behavior
stays identical to the per-scene script. Run this from the repo root,
next to run_markdown_tts.py, so the import resolves.

MODE SELECTION:
Rather than editing .env's MODEL_ID every time you want to switch between
CustomVoice, VoiceDesign, and Base (clone), use --mode as a shortcut:

    --mode custom   -> Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (needs --speaker)
    --mode design   -> Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
    --mode clone    -> Qwen/Qwen3-TTS-12Hz-1.7B-Base (needs --ref-audio + --ref-text/--ref-text-file)

--model still works as an explicit override/escape hatch if you need a
model id --mode doesn't cover (e.g. the 0.6B variants). .env's MODEL_ID
is only consulted if neither --mode nor --model is given, so you're never
forced to edit .env just to A/B between modes.

SINGLE-SCENE TESTING:
Use --only <out_stem> to render just one scene from a manifest/scene-dir
instead of the whole batch -- useful for spot-checking a mode or a fix
before committing to a full run. The model still only loads once either
way; --only just filters which scenes get looped over afterward.

Usage (matches run_test_scenes.sh, CustomVoice):
    python render_batch.py \
        --scene-dir docs/test_scenes \
        --out-dir output/voice_test \
        --manifest test_scenes_manifest.json \
        --mode custom --speaker Ryan \
        --instruct "$INSTRUCT" \
        --max-new-tokens 400 \
        --takes 2

Usage (VoiceDesign):
    python render_batch.py \
        --scene-dir docs/test_scenes \
        --out-dir output/voice_test \
        --manifest test_scenes_manifest.json \
        --mode design \
        --instruct "$INSTRUCT" \
        --max-new-tokens 400 \
        --takes 2

Usage (Base / voice clone):
    python render_batch.py \
        --scene-dir docs/test_scenes \
        --out-dir output/voice_test \
        --manifest test_scenes_manifest.json \
        --mode clone \
        --ref-audio docs/voice_reference/techbear_ref.wav \
        --ref-text-file docs/voice_reference/techbear_ref.txt \
        --max-new-tokens 400 \
        --takes 2

Usage (single scene from the full monologue, e.g. to spot-check before
committing to a 29-scene run):
    python render_batch.py \
        --scene-dir docs/monologue_scenes \
        --out-dir output/monologue \
        --manifest scenes_manifest.json \
        --mode clone \
        --max-new-tokens 1000 \
        --takes 1 \
        --only scene01

Without --manifest, renders every scene*.md file in --scene-dir,
alphabetically, using its filename stem as the output name.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import soundfile as sf
import torch
from dotenv import load_dotenv


from qwen_tts import Qwen3TTSModel
from run_markdown_tts import extract_stage_directions, markdown_to_text, parse_dtype

# Shortcut names -> actual Qwen3-TTS model ids. Update here if you move to
# a different size (e.g. 0.6B) or a newer checkpoint tag.
MODE_MODEL_IDS = {
    "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a directory of scene markdown files with one model load."
    )
    parser.add_argument("--scene-dir", required=True, help="Directory of scene*.md files")
    parser.add_argument("--out-dir", required=True, help="Directory to write wav files into")
    parser.add_argument(
        "--manifest",
        help="Optional JSON manifest: list of {file, out_stem?, instruct_suffix?}. "
             "Without this, every scene*.md in --scene-dir is rendered, alphabetically, "
             "using its filename stem as out_stem.",
    )
    parser.add_argument(
        "--only",
        help="Render only the manifest entry (or scene*.md file) whose out_stem "
             "matches this value, instead of the whole batch. Useful for "
             "spot-checking one scene before committing to a full run.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_MODEL_IDS.keys()),
        help="Shortcut for MODEL_ID: custom, design, or clone. "
             "Takes precedence over --model and .env's MODEL_ID.",
    )
    parser.add_argument("--model", help="Explicit MODEL_ID override (escape hatch beyond --mode)")
    parser.add_argument("--instruct", default="", help="Base instruct string (CustomVoice/VoiceDesign)")
    parser.add_argument("--takes", type=int, default=1, help="Number of takes to render per scene")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--language", help="Overrides LANGUAGE")
    parser.add_argument("--speaker", help="Overrides SPEAKER (CustomVoice only)")
    parser.add_argument("--ref-audio", help="Path to reference audio for voice cloning (clone mode only)")
    parser.add_argument("--ref-text", help="Transcript of --ref-audio (clone mode only)")
    parser.add_argument("--ref-text-file", help="Path to a file containing the --ref-audio transcript")
    parser.add_argument("--device", help="Overrides DEVICE")
    parser.add_argument("--dtype", help="Overrides DTYPE")
    parser.add_argument("--attn-implementation", help="Overrides ATTN_IMPLEMENTATION")
    parser.add_argument("--format", default=None, help="Overrides AUDIO_FORMAT")
    return parser.parse_args()


def load_manifest(manifest_path: str, scene_dir: Path) -> list[dict]:
    if manifest_path:
        entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        # Normalize: accept the existing scenes_manifest.json shape
        # ({"scene", "file", "direction", "wav"}) as well as a plain
        # {"file", "out_stem", "instruct_suffix"} shape.
        normalized = []
        for e in entries:
            file = e["file"]
            out_stem = e.get("out_stem") or Path(e.get("wav", file)).stem
            normalized.append({
                "file": file,
                "out_stem": out_stem,
                "instruct_suffix": e.get("instruct_suffix", ""),
            })
        return normalized
    return [
        {"file": p.name, "out_stem": p.stem, "instruct_suffix": ""}
        for p in sorted(scene_dir.glob("scene*.md"))
    ]


def resolve_model_id(args: argparse.Namespace) -> str:
    """--mode wins if given, then --model, then .env's MODEL_ID.
    Keeps .env purely as a fallback default rather than something that
    has to be edited every time you switch between comparison modes."""
    if args.mode:
        return MODE_MODEL_IDS[args.mode]
    if args.model:
        return args.model
    env_model_id = os.getenv("MODEL_ID")
    if env_model_id:
        return env_model_id
    raise SystemExit(
        "No model specified. Use --mode {custom,design,clone}, --model <id>, "
        "or set MODEL_ID in .env."
    )


def resolve_ref_text(args: argparse.Namespace) -> str | None:
    if args.ref_text:
        return args.ref_text
    if args.ref_text_file:
        return Path(args.ref_text_file).read_text(encoding="utf-8").strip()
    env_ref_text_file = os.getenv("REF_TEXT_FILE")
    if env_ref_text_file:
        return Path(env_ref_text_file).read_text(encoding="utf-8").strip()
    return os.getenv("REF_TEXT")


def main() -> None:
    load_dotenv()
    args = build_args()

    model_id = resolve_model_id(args)

    language = args.language or os.getenv("LANGUAGE", "English")
    speaker = args.speaker or os.getenv("SPEAKER", "Ryan")
    ref_audio = args.ref_audio or os.getenv("REF_AUDIO")
    ref_text = resolve_ref_text(args)
    device = args.device or os.getenv(
        "DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = parse_dtype(args.dtype or os.getenv("DTYPE", "float32"))
    attn_implementation = args.attn_implementation or os.getenv(
        "ATTN_IMPLEMENTATION", "sdpa" if device == "mps" else "eager")
    output_format = args.format or os.getenv("AUDIO_FORMAT", "wav")

    # Fail fast on missing mode-specific requirements BEFORE loading the
    # multi-GB model, not partway through a 29-scene batch.
    if "CustomVoice" in model_id and not speaker:
        raise SystemExit("CustomVoice models require --speaker or SPEAKER in .env.")
    if "Base" in model_id and (not ref_audio or not ref_text):
        raise SystemExit(
            "Base (clone) models require --ref-audio and --ref-text (or --ref-text-file), "
            "or REF_AUDIO/REF_TEXT(_FILE) in .env."
        )

    scene_dir = Path(args.scene_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = load_manifest(args.manifest, scene_dir)
    if not scenes:
        raise SystemExit(f"No scenes found in {scene_dir} (and no manifest given).")

    if args.only:
        scenes = [s for s in scenes if s["out_stem"] == args.only]
        if not scenes:
            raise SystemExit(
                f"No scene with out_stem '{args.only}' found in manifest/scene-dir."
            )

    if attn_implementation and attn_implementation.lower() in ("flash_attention_2", "flash_attn_2", "flashattn2"):
        if importlib.util.find_spec("flash_attn") is None:
            print("Warning: flash-attn is not installed. Falling back to standard attention.")
            attn_implementation = "eager"

    print(f"Loading model once: {model_id}")
    print(f"Device: {device}, dtype: {dtype}, attention: {attn_implementation}")
    tts = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    print(f"Model loaded. Rendering {len(scenes)} scene(s), {args.takes} take(s) each.\n")

    for scene in scenes:
        scene_path = scene_dir / scene["file"]
        if not scene_path.exists():
            print(f"!! Skipping missing file: {scene_path}")
            continue

        markdown_text = scene_path.read_text(encoding="utf-8")
        cleaned_markdown, stage_directions = extract_stage_directions(markdown_text)
        text = markdown_to_text(cleaned_markdown)

        merged_instruct = " ".join(
            part for part in (args.instruct.strip(), scene["instruct_suffix"].strip()) if part
        )
        if stage_directions:
            if merged_instruct:
                merged_instruct = f"{merged_instruct}. Use the following stage directions to guide tone: {stage_directions}"
            else:
                merged_instruct = f"Use the following stage directions to guide tone: {stage_directions}"

        for take in range(1, args.takes + 1):
            suffix = f"_take{take}" if args.takes > 1 else ""
            out_path = out_dir / f"{scene['out_stem']}{suffix}.{output_format}"
            print(f"=== {scene['file']} (take {take}/{args.takes}) -> {out_path.name} ===")

            if "CustomVoice" in model_id:
                wavs, sr = tts.generate_custom_voice(
                    text=text,
                    language=language,
                    speaker=speaker,
                    instruct=merged_instruct,
                    max_new_tokens=args.max_new_tokens,
                )
            elif "VoiceDesign" in model_id:
                wavs, sr = tts.generate_voice_design(
                    text=text,
                    language=language,
                    instruct=merged_instruct,
                    max_new_tokens=args.max_new_tokens,
                )
            elif "Base" in model_id:
                wavs, sr = tts.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                raise SystemExit(
                    f"Unrecognized model type in MODEL_ID ({model_id}). "
                    "Expected CustomVoice, VoiceDesign, or Base in the model id."
                )

            audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
            sf.write(str(out_path), audio, sr, format=output_format.upper())

            # The old per-scene subprocess approach couldn't accumulate
            # memory across scenes -- each process exited and the OS
            # reclaimed everything before the next one started. This
            # script keeps one process alive for the whole batch, so
            # proactively release any cached MPS memory after each
            # render rather than risk it slowly climbing over 29 scenes.
            if device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()

    print(f"\nAll renders written to {out_dir}/")


if __name__ == "__main__":
    main()
