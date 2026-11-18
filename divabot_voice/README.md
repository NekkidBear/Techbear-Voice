# DivaBot voice: VCTK shortlist → audition → RVC training set

Brief: "DivaBot Voice Identity v1.1" (Ask TechBear project). Targets in scientific pitch (middle C = C4):
speaking centre A3–B3 (220–247 Hz); shortlist VCTK women whose median speaking F0 is 200–250 Hz.

1. Download VCTK 0.92 (`VCTK-Corpus-0.92.zip`, ~11 GB) from https://datashare.ed.ac.uk/handle/10283/3443
   and unzip it to `Techbear-Voice/datasets/VCTK-Corpus-0.92/` (it holds `speaker-info.txt` and `wav48_silence_trimmed/`).
2. `pip install -r divabot_voice/requirements.txt` (and ffmpeg).
3. Rank every female speaker by pitch:
   `python divabot_voice/vctk_shortlist.py measure --vctk datasets/VCTK-Corpus-0.92 --out divabot_voice/results`
   → `results/vctk_shortlist.md` (grouped by accent) and `vctk_female_pitch.csv`.
4. Audition clips (same sentences for every speaker, loudness-matched):
   `python divabot_voice/vctk_shortlist.py audition --vctk datasets/VCTK-Corpus-0.92 --top 12 --out divabot_voice/results`
5. Your baseline for the RVC transpose: record ~1 minute reading DivaBot lines in your natural speaking voice, then
   `python divabot_voice/vctk_shortlist.py baseline that_file.wav`
6. Build the blend's training folder (minutes per speaker; 15–20 min total, 2–4 speakers from one accent group):
   `python divabot_voice/vctk_shortlist.py trainset --vctk datasets/VCTK-Corpus-0.92 --blend p228:8 p240:6 p257:6 --name divabot_v1 --out divabot_voice/results`
   → 48 kHz mono WAVs, `blend_manifest.json`, and `ATTRIBUTION.txt` (VCTK is CC BY 4.0: credit it in descriptions).

Skipped automatically: p280 and p315 (known recording problems).
