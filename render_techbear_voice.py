"""
Ask TechBear — Qwen3-TTS production render (env-driven)
Renders a given text with the speaker/instruct/device config defined in .env,
instead of hardcoding values in the script. Use this once a voice has been
picked (e.g. Ryan + the camp instruct) and you're generating real takes.

Setup:
    source backend/venv/bin/activate
    pip install torch soundfile python-dotenv --break-system-packages
    pip install git+https://github.com/QwenLM/Qwen3-TTS.git --break-system-packages

Usage:
    # render the built-in test scene 01
    python render_techbear_voice.py

    # render arbitrary text
    python render_techbear_voice.py --text "Well now, sugar. Sit down."

    # render text from a file (e.g. Techbear_Social_Intro.md content, plain text)
    python render_techbear_voice.py --text-file social_intro.txt

Expects a .env file in the working directory (or set via --env-file) with:
    MODEL_ID, LANGUAGE, SPEAKER, INSTRUCT, DEVICE, DTYPE, ATTN_IMPLEMENTATION, AUDIO_FORMAT
"""

import argparse
from pathlib import Path

import torch
import soundfile as sf
from dotenv import dotenv_values
from qwen_tts import Qwen3TTSModel

TEST_SCENE_01 = (
    "Well now, sugar. "
    "Take a deep breath. "
    "Your printer isn't possessed. "
    "It's just angry. "
    "And frankly? "
    "I understand."
)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_config(env_file: str) -> dict:
    cfg = dotenv_values(env_file)
    required = ["MODEL_ID", "LANGUAGE", "SPEAKER", "DEVICE", "DTYPE"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"Missing required .env keys: {missing}")
    return cfg


def resolve_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    return TEST_SCENE_01


def main():
    parser = argparse.ArgumentParser(description="Render TechBear voice via Qwen3-TTS")
    parser.add_argument("--env-file", default=".env", help="Path to .env config (default: ./.env)")
    parser.add_argument("--text", help="Text to render")
    parser.add_argument("--text-file", help="Path to a text file to render")
    parser.add_argument("--out", default=None, help="Output wav path (default: auto-named)")
    args = parser.parse_args()

    cfg = load_config(args.env_file)
    text = resolve_text(args)

    dtype = DTYPE_MAP.get(cfg["DTYPE"], torch.float32)
    device = cfg["DEVICE"]
    attn_impl = cfg.get("ATTN_IMPLEMENTATION", "sdpa")
    audio_format = cfg.get("AUDIO_FORMAT", "wav")

    print(f"Loading {cfg['MODEL_ID']} on {device} ({cfg['DTYPE']}, attn={attn_impl}) ...")
    model = Qwen3TTSModel.from_pretrained(
        cfg["MODEL_ID"],
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    print(f"Speaker: {cfg['SPEAKER']} | Language: {cfg['LANGUAGE']}")
    print(f"Instruct: {cfg.get('INSTRUCT', '(none)')[:80]}...")
    print(f"Text: {text[:80]}...")

    wavs, sr = model.generate_custom_voice(
        text=text,
        language=cfg["LANGUAGE"],
        speaker=cfg["SPEAKER"],
        instruct=cfg.get("INSTRUCT", ""),
    )

    out_path = Path(args.out) if args.out else Path(f"{cfg['SPEAKER'].lower()}_render.{audio_format}")
    sf.write(out_path, wavs[0], sr)
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
