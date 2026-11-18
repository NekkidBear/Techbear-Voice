#!/usr/bin/env python3
"""DivaBot voice: shortlist VCTK speakers by pitch, build audition and training sets.

Targets come from "DivaBot Voice Identity v1.1" (scientific pitch, middle C = C4):
speaking centre A3-B3 (220-247 Hz), shortlist band ~200-250 Hz median F0.

Usage
  python vctk_shortlist.py measure  --vctk VCTK-Corpus-0.92 [--clips 40] [--out results]
  python vctk_shortlist.py audition --vctk VCTK-Corpus-0.92 [--top 12] [--out results]
  python vctk_shortlist.py baseline your_reading.wav [more.wav ...]
  python vctk_shortlist.py trainset --vctk VCTK-Corpus-0.92 --blend p228:8 p240:6 p257:6 \
         --name divabot_v1 [--out results]   (minutes per speaker)

The VCTK folder is the unzipped 0.92 release: it holds speaker-info.txt and
wav48_silence_trimmed/. VCTK 0.92 is CC BY 4.0: credit it (see ATTRIBUTION.txt).
"""
import argparse, csv, json, os, random, shutil, subprocess, sys
from pathlib import Path

import numpy as np

TARGET_LO, TARGET_HI, TARGET_MID = 200.0, 250.0, 233.0  # Hz
SKIP = {"p280", "p315"}  # known recording problems in the corpus
ATTRIBUTION = (
    "DivaBot's voice model was trained in part on the CSTR VCTK Corpus (v0.92), "
    "Centre for Speech Technology Research, University of Edinburgh, licensed "
    "CC BY 4.0. https://doi.org/10.7488/ds/2645\n"
)


def read_speakers(vctk: Path):
    rows = []
    with open(vctk / "speaker-info.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].upper() == "ID":
                continue
            sid = parts[0] if parts[0].startswith(("p", "s")) else "p" + parts[0]
            rows.append({
                "id": sid,
                "age": parts[1] if len(parts) > 1 else "",
                "gender": parts[2] if len(parts) > 2 else "",
                "accent": parts[3] if len(parts) > 3 else "",
                "region": " ".join(parts[4:]),
            })
    return rows


def clips_for(vctk: Path, sid: str):
    d = vctk / "wav48_silence_trimmed" / sid  # 0.92 layout
    if not d.is_dir():
        d = vctk / "wav48" / sid  # older 0.80 layout (Kaggle mirrors); note its different licence
    if not d.is_dir():
        return []
    files = sorted(d.glob("*_mic1.flac")) or sorted(d.glob("*.flac")) or sorted(d.glob("*.wav"))
    return files


def f0_stats(paths):
    import librosa
    vals = []
    for p in paths:
        y, sr = librosa.load(p, sr=16000, mono=True)
        f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=700, sr=sr, frame_length=1024)
        f0 = f0[voiced & ~np.isnan(f0)]
        vals.append(f0)
    f0 = np.concatenate(vals) if vals else np.array([])
    if f0.size < 50:
        return None
    p10, med, p90 = np.percentile(f0, [10, 50, 90])
    return {
        "median_hz": round(float(med), 1),
        "p10_hz": round(float(p10), 1),
        "p90_hz": round(float(p90), 1),
        "range_st": round(float(12 * np.log2(p90 / p10)), 1),
        "median_note": librosa.hz_to_note(med),
        "dist_st": round(float(abs(12 * np.log2(med / TARGET_MID))), 2),
    }


def cmd_measure(a):
    vctk, out = Path(a.vctk), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    results = []
    females = [s for s in read_speakers(vctk) if s["gender"].upper().startswith("F") and s["id"] not in SKIP]
    for i, s in enumerate(females, 1):
        files = clips_for(vctk, s["id"])
        if not files:
            print(f"[{i}/{len(females)}] {s['id']}: no audio, skipped")
            continue
        sample = rng.sample(files, min(a.clips, len(files)))
        st = f0_stats(sample)
        if not st:
            continue
        st["in_band"] = TARGET_LO <= st["median_hz"] <= TARGET_HI
        results.append({**s, **st, "clips_measured": len(sample)})
        print(f"[{i}/{len(females)}] {s['id']} {s['accent']}/{s['region']}: median {st['median_hz']} Hz ({st['median_note']})")
    results.sort(key=lambda r: (not r["in_band"], r["dist_st"]))
    keys = ["id", "age", "accent", "region", "median_hz", "median_note", "p10_hz", "p90_hz",
            "range_st", "dist_st", "in_band", "clips_measured"]
    with open(out / "vctk_female_pitch.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(results)
    write_summary(results, out / "vctk_shortlist.md")
    print(f"\nWrote {out/'vctk_female_pitch.csv'} and {out/'vctk_shortlist.md'}")


def write_summary(results, path):
    lines = ["# VCTK female speakers ranked for DivaBot", "",
             f"Shortlist band: median {TARGET_LO:.0f}-{TARGET_HI:.0f} Hz; ranked by distance from {TARGET_MID:.0f} Hz.",
             "Group your blend within one accent group so vowels stay clean.", ""]
    by_accent = {}
    for r in results:
        by_accent.setdefault(r["accent"] or "Unknown", []).append(r)
    for acc, rs in sorted(by_accent.items(), key=lambda kv: -sum(x["in_band"] for x in kv[1])):
        lines += [f"## {acc} ({sum(x['in_band'] for x in rs)} in band)", "",
                  "| ID | Age | Region | Median | Note | 10th–90th pct | Range (st) | In band |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in rs:
            lines.append(f"| {r['id']} | {r['age']} | {r['region']} | {r['median_hz']} Hz | {r['median_note']} | "
                         f"{r['p10_hz']}–{r['p90_hz']} Hz | {r['range_st']} | {'yes' if r['in_band'] else ''} |")
        lines.append("")
    Path(path).write_text("\n".join(lines))


def ffmpeg_concat(files, dest, gap=0.4):
    dest.parent.mkdir(parents=True, exist_ok=True)
    inputs, filt = [], []
    for i, f in enumerate(files):
        inputs += ["-i", str(f)]
        filt.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=mono,apad=pad_dur={gap}[a{i}]")
    chain = "".join(f"[a{i}]" for i in range(len(files)))
    fc = ";".join(filt) + f";{chain}concat=n={len(files)}:v=0:a=1,loudnorm=I=-18:TP=-2[out]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", fc,
                    "-map", "[out]", "-ar", "48000", str(dest)], check=True)


def cmd_audition(a):
    vctk, out = Path(a.vctk), Path(a.out)
    rows = list(csv.DictReader(open(out / "vctk_female_pitch.csv")))[: a.top]
    # Same sentences for everyone: VCTK's shared texts (001 elicitation, then the rainbow passage)
    for rank, r in enumerate(rows, 1):
        files = clips_for(vctk, r["id"])
        by_num = {f.name.split("_")[1]: f for f in files}
        pick = [by_num[n] for n in ("001", "002", "003", "004", "005", "006") if n in by_num] or files[:6]
        name = f"{rank:02d}_{r['id']}_{r['accent']}_{float(r['median_hz']):.0f}Hz.wav".replace(" ", "")
        ffmpeg_concat(pick, out / "audition" / name)
        print("wrote", name)


def cmd_baseline(a):
    st = f0_stats([Path(p) for p in a.files])
    if not st:
        sys.exit("Not enough voiced speech to measure.")
    med = st["median_hz"]
    print(json.dumps(st, indent=2))
    for t in (220, 233, 247):
        print(f"transpose to {t} Hz: {12*np.log2(t/med):+.1f} semitones")


def cmd_trainset(a):
    vctk, out = Path(a.vctk), Path(a.out) / "training" / a.name
    out.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    rng, manifest = random.Random(11), []
    for item in a.blend:
        sid, mins = item.split(":")
        files = clips_for(vctk, sid)
        rng.shuffle(files)
        total, used = 0.0, 0
        for f in files:
            if total >= float(mins) * 60:
                break
            dest = out / f"{f.stem}.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(f), "-ac", "1", "-ar", "48000",
                            str(dest)], check=True)
            total += sf.info(str(dest)).duration
            used += 1
        manifest.append({"speaker": sid, "minutes": round(total / 60, 2), "clips": used})
        print(f"{sid}: {total/60:.1f} min from {used} clips")
    (out / "blend_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "ATTRIBUTION.txt").write_text(ATTRIBUTION + "Speakers: " + ", ".join(m["speaker"] for m in manifest) + "\n")
    print("Training folder:", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure"); m.add_argument("--vctk", required=True); m.add_argument("--clips", type=int, default=40); m.add_argument("--out", default="results")
    au = sub.add_parser("audition"); au.add_argument("--vctk", required=True); au.add_argument("--top", type=int, default=12); au.add_argument("--out", default="results")
    b = sub.add_parser("baseline"); b.add_argument("files", nargs="+")
    t = sub.add_parser("trainset"); t.add_argument("--vctk", required=True); t.add_argument("--blend", nargs="+", required=True); t.add_argument("--name", required=True); t.add_argument("--out", default="results")
    a = ap.parse_args()
    {"measure": cmd_measure, "audition": cmd_audition, "baseline": cmd_baseline, "trainset": cmd_trainset}[a.cmd](a)


if __name__ == "__main__":
    main()
