# Qwen3 TTS Workspace (local, Mac-friendly)

Local markdown-to-speech wrapper around the official `qwen-tts` PyPI package.
No separate repo clone needed — `qwen-tts` installs straight from PyPI.

## Setup (macOS)

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -U pip
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and adjust if needed:

   ```bash
   cp .env.example .env
   ```

   Defaults are already set for Mac: `DEVICE=mps` (Apple Silicon GPU),
   `DTYPE=float16`, `ATTN_IMPLEMENTATION=sdpa`. Don't set `DEVICE=cuda:0` or
   `ATTN_IMPLEMENTATION=flash_attention_2` — those are NVIDIA-only and will
   fail on a Mac. If you're on an Intel Mac with no discrete GPU, set
   `DEVICE=cpu` instead (generation will just be slower).

## Usage

```bash
python run_markdown_tts.py --input docs/example.md --output output/example.wav
```

Override any `.env` value on the command line:

```bash
python run_markdown_tts.py --input docs/example.md --output output/example.wav --instruct "Speak in a calm narrative voice"
```

## Notes

- Use a `CustomVoice` or `VoiceDesign` model in `MODEL_ID` for text-only markdown generation. `Base` models require a reference audio prompt and aren't supported by this wrapper.
- The first run will download the model weights from Hugging Face (a few GB) — that's expected and only happens once.
- This workspace does not launch the Gradio demo or any port-based server.
- The `src/` folder and `package.json` in this project are a **separate, unrelated implementation** — a Node/TypeScript CLI that calls a remote Qwen cloud API instead of running the model locally. They're leftover from Copilot generating two different approaches at once. Since you're going the local-model route, you can safely delete `src/`, `package.json`, `tsconfig.json`, and `node_modules/` (if present) to declutter — they share no code with the Python path.
