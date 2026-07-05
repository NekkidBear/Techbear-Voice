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

Usage (matches run_test_scenes.sh):
    python render_batch.py \
        --scene-dir docs/test_scenes \
        --out-dir output/voice_test \
        --manifest test_scenes_manifest.json \
        --instruct "$BASE_VOICE" \
        --max-new-tokens 400 \
        --takes 2

Usage (matches run_full_monologue.sh):
    python render_batch.py \
        --scene-dir docs/monologue_scenes \
        --out-dir output/monologue \
        --manifest scenes_manifest.json \
        --instruct "$BASE_VOICE" \
        --max-new-tokens 1000 \
        --takes 1

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
    parser.add_argument("--instruct", default="", help="Base instruct string (e.g. BASE_VOICE)")
    parser.add_argument("--takes", type=int, default=1, help="Number of takes to render per scene")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--model", help="Overrides MODEL_ID")
    parser.add_argument("--language", help="Overrides LANGUAGE")
    parser.add_argument("--speaker", help="Overrides SPEAKER")
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


def main() -> None:
    load_dotenv()
    args = build_args()

    model_id = args.model or os.getenv("MODEL_ID")
    if not model_id:
        raise SystemExit("MODEL_ID is required either in .env or via --model.")

    language = args.language or os.getenv("LANGUAGE", "English")
    speaker = args.speaker or os.getenv("SPEAKER", "Ryan")
    device = args.device or os.getenv(
        "DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = parse_dtype(args.dtype or os.getenv("DTYPE", "float32"))
    attn_implementation = args.attn_implementation or os.getenv(
        "ATTN_IMPLEMENTATION", "sdpa" if device == "mps" else "eager")
    output_format = args.format or os.getenv("AUDIO_FORMAT", "wav")

    scene_dir = Path(args.scene_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = load_manifest(args.manifest, scene_dir)
    if not scenes:
        raise SystemExit(f"No scenes found in {scene_dir} (and no manifest given).")

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
            else:
                raise SystemExit(
                    "Base models require a voice clone prompt and are not supported here. "
                    "Use a CustomVoice or VoiceDesign model id."
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
