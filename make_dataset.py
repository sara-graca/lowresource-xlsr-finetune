"""
USAGE
-----
1. Extract sentence clips:
    python make_dataset.py create_audio \
        --wav_dir   /path/to/audio \
        --trans_dir /path/to/annotations \
        --out_dir   /path/to/output \
        [--language tdh]

   --language is optional (ISO 639-3 code). 
   Without it, only a general cleanup runs.
   With a code, a language-specific normalizer runs on top:

     tdh (Thulung) 
     nru (Na)       also drops "fin peu audible" and "BEGAIMENT" clips

   To add a language, define a normalize_<code> and register it in NORMALIZERS.

2. Split into train/valid/test:
    python make_dataset.py create_dataset \
        --path /path/to/output/
"""

import argparse
import csv
import re
import xml.etree.ElementTree as et
from pathlib import Path
from unicodedata import normalize

import numpy as np
import pandas as pd
import torch
import torchaudio


def extract_information(xml_file):
    information = {}
    skipped = 0
    root = et.parse(xml_file).getroot()
    nodes = root.findall("S") or root.findall("W")
    for child in nodes:
        sent_id = child.attrib.get("id")
        form = child.find("FORM")
        audio = child.find("AUDIO")
        if form is None or form.text is None or audio is None:
            skipped += 1
            continue
        timecode = audio.attrib
        if "start" not in timecode or "end" not in timecode:
            skipped += 1
            continue
        information[sent_id] = [form.text, timecode["start"], timecode["end"]]
    if skipped:
        print(f"  [WARN] {Path(xml_file).name}: skipped {skipped} sentence(s) "
              f"without timecode or text")
    return information


def clean_markup(text):
    """Language-general cleanup."""
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[,;!?]', '', text)
    text = normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_tdh(text):
    """Thulung-specific."""
    text = re.sub(r'--+', ' ', text)
    text = text.replace('-', '')
    text = re.sub(r'[…]+', ' ', text)
    text = re.sub(r'\.+', ' ', text)
    text = text.replace(':', 'ː')
    text = text.replace('᷉', '̃')
    text = text.replace('̵', '')
    text = re.sub(r'["=_/§X]', '', text)
    return clean_markup(text)


def normalize_nru(text):
    """Yongning Na-specific."""
    text = text.replace("...", "…")
    text = text.replace("◊", "|")
    text = text.replace("F", "")
    text = text.replace("D", "")
    text = text.replace("<", "")
    text = text.replace(">", "")
    text = re.sub(r'm+…', 'mmm…', text)
    text = re.sub(r'ʰ+', 'ʰ', text)
    return clean_markup(text)


# ISO 639-3 code -> normalizer. To add a language, write a normalize_<code>
# that ends by calling clean_markup, and add one entry here.
NORMALIZERS = {
    "tdh": normalize_tdh,
    "nru": normalize_nru,
}


def get_normalizer(language):
    """Return the normalizer for an ISO 639-3 code, or the general cleanup
    if no language is given or the code is unknown."""
    if language is None:
        return clean_markup
    fn = NORMALIZERS.get(language)
    if fn is None:
        print(f"  [WARN] no normalizer for '{language}', using general cleanup only")
        return clean_markup
    return fn


def skip_nru(text):
    """Na: drop clips marked as barely audible or stuttered."""
    return "fin peu audible" in text or "BEGAIEMENT" in text

SKIP_FILTERS = {
    "nru": skip_nru,
}

def get_skip_filter(language):
    """Return the clip-skipping predicate for an ISO 639-3 code, or a no-op
    (drop nothing) if the language has no filter."""
    return SKIP_FILTERS.get(language, lambda text: False)


def create_audio_tsv(args):
    wav_dir = Path(args.wav_dir)
    trans_dir = Path(args.trans_dir)
    out_dir = Path(args.out_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    normalizer = get_normalizer(args.language)
    skip = get_skip_filter(args.language)

    wav_files = sorted(f for f in wav_dir.iterdir()
                       if f.suffix.lower() == ".wav" and not f.name.startswith("_tmp"))
    print(f"{len(wav_files)} wav files found in {wav_dir}")

    tsv_out = open(out_dir / "all.tsv", "wt")
    writer = csv.writer(tsv_out, delimiter="\t")
    writer.writerow(["path", "sentence"])

    total_dur = 0.0
    problem = ""
    xml_missing = []

    for wav in wav_files:
        xml = trans_dir / wav.with_suffix(".xml").name
        if not xml.exists():
            candidates = [x for x in trans_dir.glob("*.xml")
                          if x.stem.lower() == wav.stem.lower()]
            xml = candidates[0] if candidates else None

        if xml is None:
            xml_missing.append(wav.name)
            continue

        try:
            info = extract_information(str(xml))
            if not info:
                print(f"  [WARN] empty annotation: {xml.name}")
                continue

            waveform, sr = torchaudio.load(str(wav))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)
            if sr != 16_000:
                waveform = torchaudio.transforms.Resample(sr, 16_000)(waveform)
                sr = 16_000

            for transcript, (start, end, sent_id) in info.items():
                if skip(transcript):
                    continue
                transcript = normalizer(transcript)
                if not transcript:
                    continue
                dur = float(end) - float(start)
                if dur <= 1 or dur > 20:
                    continue

                start_s = int(float(start) * sr)
                end_s = min(int((float(end) + 0.2) * sr), waveform.shape[1])
                clip = waveform[:, start_s:end_s]

                clip_name = f"{wav.stem}_{sent_id}.wav"
                torchaudio.save(str(clips_dir / clip_name), clip, sr)
                writer.writerow([clip_name, transcript])
                total_dur += dur

        except Exception as e:
            problem += f"  {wav.name}: {e}\n"

    tsv_out.close()

    if xml_missing:
        print(f"No XML found for {len(xml_missing)} file(s): {xml_missing}")
    if problem:
        print(f"Errors:\n{problem}")
    print(f"{total_dur/60:.1f} min of audio prepared  →  {out_dir / 'all.tsv'}")


def create_dataset(args):
    """Split into train/valid/test (80/10/10) at the recording level, so all
    clips from one recording stay in the same split (no data leakage)."""
    path = args.path

    corpus = pd.read_csv(path + 'all.tsv', sep='\t')

    corpus["_recording"] = corpus["path"].apply(
        lambda p: re.sub(r"_S\d+.*\.wav$", "", p)
    )

    recordings = sorted(corpus["_recording"].unique())
    rng = np.random.default_rng(seed=42)
    rng.shuffle(recordings)

    n = len(recordings)
    b = int(n * 0.1)          # valid and test get 10% each
    a = n - 2 * b             # train gets the rest
    train_recs = set(recordings[:a])
    valid_recs = set(recordings[a:a + b])
    test_recs = set(recordings[a + b:])

    def split_name(rec):
        if rec in train_recs: return "train"
        if rec in valid_recs: return "valid"
        return "test"

    corpus["_split"] = corpus["_recording"].apply(split_name)

    train = corpus[corpus["_split"] == "train"].drop(columns=["_recording", "_split"])
    val = corpus[corpus["_split"] == "valid"].drop(columns=["_recording", "_split"])
    test = corpus[corpus["_split"] == "test"].drop(columns=["_recording", "_split"])

    train.to_csv(path + 'train.tsv', index=False, sep='\t')
    val.to_csv(path + 'valid.tsv', index=False, sep='\t')
    test.to_csv(path + 'test.tsv', index=False, sep='\t')

    print(f"Recording-level split ({n} recordings total, seed=42):")
    print(f"  train: {len(train_recs)} recordings  →  {len(train)} clips")
    print(f"  valid: {len(valid_recs)} recordings  →  {len(val)} clips")
    print(f"  test:  {len(test_recs)}  recordings  →  {len(test)} clips")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    create_audio = subparsers.add_parser(
        "create_audio",
        help="Extract sentence-level clips from wav files using XML timecodes.")
    create_audio.add_argument("--wav_dir", type=str, required=True,
                              help="Directory containing source wav files.")
    create_audio.add_argument("--trans_dir", type=str, required=True,
                              help="Directory containing XML transcription files.")
    create_audio.add_argument("--out_dir", type=str, required=True,
                              help="Output directory for clips/ and all.tsv.")
    create_audio.add_argument("--language", type=str, default=None,
                              help="Optional ISO 639-3 code (e.g. tdh). Selects a "
                                   "language-specific normalizer; omit for general cleanup only.")
    create_audio.set_defaults(func=create_audio_tsv)

    split_dataset = subparsers.add_parser("create_dataset",
                                          help="Create dataset - train/val/test tsv files.")
    split_dataset.add_argument('--path', required=True, help="path of the corpus with wav and transcription files")
    split_dataset.set_defaults(func=create_dataset)

    args = parser.parse_args()
    args.func(args)