"""
Ask TechBear — Qwen3-TTS Voice Comparison Test (3-way, 27 clips)
Runs "test scene 01" across all 9 official Qwen3-TTS CustomVoice presets,
each in three takes:
  1. neutral   — no instruct, raw preset timbre
  2. raspy     — generic mellow/smoker instruct only, isolating the vocal
                 quality trait across all speakers
  3. tb_rasp   — full TechBear character instruct (Fierstein/Lynde/RuPaul
                 camp) with the rasp/vocal-fry quality pushed harder, to
                 see how each speaker's raw rasp potential holds up once
                 the full characterization is layered on top

9 speakers x 3 variants = 27 clips.

Setup:
    source backend/venv/bin/activate
    pip install torch soundfile --break-system-packages
    pip install git+https://github.com/QwenLM/Qwen3-TTS.git --break-system-packages

Usage:
    python voice_test_scene01_3way.py

Output:
    voice_test_scene01/<speaker>_neutral.wav
    voice_test_scene01/<speaker>_raspy.wav
    voice_test_scene01/<speaker>_tb_rasp.wav
"""

from pathlib import Path
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUTPUT_DIR = Path("voice_test_scene01")
OUTPUT_DIR.mkdir(exist_ok=True)

# All 9 official Qwen3-TTS CustomVoice presets
SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]

TEST_SCENE_01 = (
    "Well now, sugar. "
    "Take a deep breath. "
    "Your printer isn't possessed. "
    "It's just angry. "
    "And frankly? "
    "I understand."
)

# Three instruct conditions to compare per speaker
INSTRUCT_VARIANTS = {
    "neutral": "",
    "raspy": (
        "Speak low and mellow, unhurried, with a slight rasp and vocal fry "
        "like someone who smoked for years — warm and worn, not harsh or "
        "aggressive, not young or bright."
    ),
    "tb_rasp": (
        "Middle-aged male voice, mid-American accent with Southern warmth. "
        "Speak low and mellow, unhurried, with a pronounced rasp and vocal "
        "fry — like someone who smoked for years — warm and worn beneath "
        "the glamour. Harvey Fierstein-style grand theatrical warmth and "
        "total confidence. Nasal, high-camp Broadway-diva delivery: quick "
        "Paul Lynde-style arched-eyebrow zingers, RuPaul-level showmanship "
        "and authority, sudden theatrical gasps and dramatic pauses. Full "
        "vocal range, not flat — swings between dry regal deadpan and big "
        "exuberant camp flourish. Warm Dolly Parton sincerity underneath "
        "the glamour. Sarcastic edge, never mean — roasts the problem, not "
        "the person."
    ),
}


def pick_device_and_dtype():
    """Pick a working device/dtype combo. bfloat16 is safest on CUDA;
    MPS/CPU are more reliable with float32."""
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def main():
    device, dtype = pick_device_and_dtype()
    print(f"Loading {MODEL_ID} on {device} ({dtype}) ...")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map=device,
        dtype=dtype,
    )

    total_clips = len(SPEAKERS) * len(INSTRUCT_VARIANTS)
    print(f"Planning {len(SPEAKERS)} speakers x {len(INSTRUCT_VARIANTS)} variants = {total_clips} clips")

    for variant_name, instruct_text in INSTRUCT_VARIANTS.items():
        print(f"\n--- Generating '{variant_name}' takes for {len(SPEAKERS)} speakers ---")

        texts = [TEST_SCENE_01] * len(SPEAKERS)
        languages = ["English"] * len(SPEAKERS)
        instructs = [instruct_text] * len(SPEAKERS)

        wavs, sr = model.generate_custom_voice(
            text=texts,
            language=languages,
            speaker=SPEAKERS,
            instruct=instructs,
        )

        for speaker, wav in zip(SPEAKERS, wavs):
            out_path = OUTPUT_DIR / f"{speaker.lower()}_{variant_name}.wav"
            sf.write(out_path, wav, sr)
            print(f"  saved {out_path}")

    print(f"\nDone. All {total_clips} clips are in:", OUTPUT_DIR.resolve())
    print("Suggested listening pass: neutral -> raspy -> tb_rasp per speaker,")
    print("so you hear how much of the rasp survives once camp is layered on.")


if __name__ == "__main__":
    main()
