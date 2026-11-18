#!/usr/bin/env python3
"""Prepare Jason's raw DivaBot performances for analysis and RVC conversion.

For each input WAV: mono, 48 kHz, 24-bit; 70 Hz high-pass; leading/trailing
silence trimmed; long internal pauses capped (default 1.2 s). Writes a manifest
with each clip's length, median pitch and the RVC transpose that would land it on
DivaBot's 233 Hz speaking centre (brief: "DivaBot Voice Identity v1.1").

Usage
  python prep_performances.py --name ep01 "/path/to/clips/ep01_DB*.wav"
  -> divabot_voice/performances/ep01/*.wav + manifest.csv

Audio stays out of git (*.wav is ignored); the manifest is small and tracked.
"""
import argparse, csv, glob
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfiltfilt

SR, TARGET_HZ = 48000, 233.0


def cap_pauses(y, sr, top_db, max_pause):
    """Trim the ends and shorten any pause longer than max_pause seconds."""
    iv = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
    if len(iv) == 0:
        return y[:0]
    pad, cap = int(0.08 * sr), int(max_pause * sr)
    pieces, prev_end = [], None
    for s, e in iv:
        s, e = max(0, s - pad), min(len(y), e + pad)
        if prev_end is not None and s > prev_end:
            gap = s - prev_end
            pieces.append(y[s - min(gap, cap):s])  # keep up to `cap` of real room tone
        elif prev_end is not None and s < prev_end:
            s = prev_end
        pieces.append(y[s:e])
        prev_end = e
    return np.concatenate(pieces)


def median_f0(y, sr):
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    f0, v, _ = librosa.pyin(y16, fmin=60, fmax=500, sr=16000, frame_length=1024)
    f0 = f0[v & ~np.isnan(f0)]
    return float(np.median(f0)) if f0.size >= 20 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="WAV files or quoted globs")
    ap.add_argument("--name", required=True, help="set name, e.g. ep01")
    ap.add_argument("--out", default=str(Path(__file__).parent / "performances"))
    ap.add_argument("--top-db", type=float, default=40)
    ap.add_argument("--max-pause", type=float, default=1.2)
    a = ap.parse_args()

    files = sorted({f for p in a.inputs for f in (glob.glob(p) or [p])})
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    sos = butter(4, 70, btype="highpass", fs=SR, output="sos")
    rows = []
    for f in files:
        y, sr = librosa.load(f, sr=SR, mono=True)
        raw_len = len(y) / SR
        y = sosfiltfilt(sos, y)
        y = cap_pauses(y, SR, a.top_db, a.max_pause)
        if len(y) < SR * 0.3:
            print(f"skip {Path(f).name}: no speech found")
            continue
        peak = np.max(np.abs(y))
        if peak > 0.98:
            y = y * (0.98 / peak)
        dest = out / Path(f).name
        sf.write(dest, y, SR, subtype="PCM_24")
        med = median_f0(y, SR)
        tr = round(12 * np.log2(TARGET_HZ / med), 1) if med else ""
        rows.append({"file": dest.name, "source": Path(f).name, "raw_s": round(raw_len, 1),
                     "clean_s": round(len(y) / SR, 1), "median_hz": round(med, 1) if med else "",
                     "note": librosa.hz_to_note(med) if med else "", "transpose_to_233": tr})
        print(f"{dest.name}: {raw_len:.1f}s -> {len(y)/SR:.1f}s, " + (f"median {med:.1f} Hz, transpose {tr:+}" if med else "no clear pitch (breath, raspberry or very short)"))
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    total = sum(r["clean_s"] for r in rows)
    print(f"\n{len(rows)} clips, {total:.1f}s of cleaned audio -> {out}")


if __name__ == "__main__":
    main()
