#!/usr/bin/env python3
import argparse
import importlib.util
import os
import re
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from markdown import markdown
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


def markdown_to_text(markdown_text: str) -> str:
    html = markdown(markdown_text, output_format="html5")
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(soup.stripped_strings)


def extract_stage_directions(markdown_text: str) -> tuple[str, str]:
    directions = []

    def replace(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        directions.append(content)
        return ""

    cleaned = re.sub(r"_\(([^)]*?)\)_", replace, markdown_text)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, " ".join(directions)


def parse_dtype(dtype_name: str) -> torch.dtype:
    normalized = (dtype_name or "float32").strip().lower()
    if normalized in ("bfloat16", "bf16"):
        return torch.bfloat16
    if normalized in ("float16", "fp16", "half"):
        return torch.float16
    if normalized in ("float32", "fp32"):
        return torch.float32
    raise ValueError("Unsupported dtype. Use bfloat16, float16, or float32.")


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert markdown into speech using a local Qwen3-TTS source package."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Markdown input file path")
    parser.add_argument("--output", "-o", required=True,
                        help="Output audio file path")
    parser.add_argument(
        "--model", help="Model id or local model path (overrides MODEL_ID)")
    parser.add_argument("--language", help="Language for generation")
    parser.add_argument(
        "--speaker", help="Speaker name for CustomVoice models")
    parser.add_argument(
        "--instruct", help="Text instruction / voice prompt override")
    parser.add_argument(
        "--device", help="Device for model loading (cpu, cuda:0, etc.)")
    parser.add_argument("--dtype", help="Torch dtype for model loading")
    parser.add_argument("--attn-implementation",
                        help="Attention implementation override")
    parser.add_argument("--format", default=None,
                        help="Output audio format (wav, flac, mp3)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Optional max_new_tokens for generation")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = build_args()

    model_id = args.model or os.getenv("MODEL_ID")
    if not model_id:
        raise SystemExit("MODEL_ID is required either in .env or via --model.")

    language = args.language or os.getenv("LANGUAGE", "English")
    speaker = args.speaker or os.getenv("SPEAKER", "Ryan")
    instruct = args.instruct or os.getenv("INSTRUCT", "")
    device = args.device or os.getenv(
        "DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = parse_dtype(args.dtype or os.getenv("DTYPE", "float32"))
    attn_implementation = args.attn_implementation or os.getenv(
        "ATTN_IMPLEMENTATION", "sdpa" if device == 'mps' else 'eager')
    output_format = args.format or os.getenv("AUDIO_FORMAT", "wav")
    max_new_tokens = args.max_new_tokens

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input markdown file not found: {input_path}")

    markdown_text = input_path.read_text(encoding="utf-8")
    cleaned_markdown, stage_directions = extract_stage_directions(
        markdown_text)
    text = markdown_to_text(cleaned_markdown)

    merged_instruct = instruct.strip()
    if stage_directions:
        if merged_instruct:
            merged_instruct = f"{merged_instruct}. Use the following stage directions to guide tone: {stage_directions}"
        else:
            merged_instruct = f"Use the following stage directions to guide tone: {stage_directions}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_id}")
    print(
        f"Device: {device}, dtype: {dtype}, attention: {attn_implementation}")
    if stage_directions:
        print(
            f"Stage directions loaded into instruction prompt: {stage_directions[:120]}{'...' if len(stage_directions) > 120 else ''}")

    if attn_implementation and attn_implementation.lower() in ("flash_attention_2", "flash_attn_2", "flashattn2"):
        flash_installed = importlib.util.find_spec("flash_attn") is not None
        if not flash_installed:
            print(
                "Warning: flash-attn is not installed. Falling back to standard attention.")
            attn_implementation = "eager"

    tts = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )

    if "CustomVoice" in model_id:
        wavs, sr = tts.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=merged_instruct,
            max_new_tokens=max_new_tokens,
        )
    elif "VoiceDesign" in model_id:
        wavs, sr = tts.generate_voice_design(
            text=text,
            language=language,
            instruct=merged_instruct,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise SystemExit(
            "Base models require a voice clone prompt and are not supported by this markdown wrapper. "
            "Use a CustomVoice or VoiceDesign model id."
        )

    audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    sf.write(str(output_path), audio, sr)
    print(f"Saved speech to {output_path}")


if __name__ == "__main__":
    main()
