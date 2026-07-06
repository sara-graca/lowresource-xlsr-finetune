# ASR Pipeline for Low-Resource Languages

A toolkit for building automatic speech recognition (ASR) models for
under-documented languages, built around fine-tuning XLS-R. It covers
the full workflow: turning raw recordings + annotations into a clean training
dataset, fine-tuning the model, and running inference on new audio with output
formats ready for linguistic annotation tools.

## Pipeline overview

```
raw audio + annotations  ─▶  make_dataset.py        ─▶  train/valid/test TSVs + clips
train/valid/test TSVs    ─▶  fine_tune_xlsr_wav2vec2.py  ─▶  fine-tuned model
new audio + model        ─▶  prediction_wav2vec2.py  ─▶  .tsv / .xml / .eaf / .txt transcripts
```

## Requirements

- Python 3.8+
- Core: `torch`, `torchaudio`, `transformers`, `datasets`, `evaluate`
- Data handling: `numpy`, `pandas`
- Audio: `librosa`, `soundfile`, `pydub` (silence-based chunking in `prediction_wav2vec2.py`)

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

---

## 1. `make_dataset.py` — Build a training dataset

Extracts sentence-level audio clips from recordings + their annotations, and
splits them into train/valid/test sets.

### Step 1: Extract sentence clips

```bash
python make_dataset.py create_audio \
    --wav_dir   /path/to/audio \
    --trans_dir /path/to/annotations \
    --out_dir   /path/to/output \
    [--language tdh]
```

- `--language` is optional (ISO 639-3 code).
  - Without it, only a general cleanup runs.
  - With a code, a language-specific normalizer also runs on top:
    - `tdh` (Thulung)
    - `nru` (Na) 

**Adding a new language:** define a `normalize_<code>` function and register
it in `NORMALIZERS`.

### Step 2: Split into train/valid/test

```bash
python make_dataset.py create_dataset \
    --path /path/to/output/
```

---

## 2. `fine_tune_xlsr_wav2vec2.py` — Fine-tune the model

```bash
python fine_tune_xlsr_wav2vec2.py \
    --train_tsv  /path/to/train.tsv \
    --val_tsv    /path/to/val.tsv \
    --output_dir /path/to/output \
    --clips_dir  /path/to/clips \
    --base_model /path/to/other/model   # optional, defaults to facebook/wav2vec2-large-xlsr-53
```

If `output_dir` already contains checkpoints, training resumes automatically.

### Input format

TSV files with two tab-separated columns, **no header**:

| column     | description                                  |
|------------|-----------------------------------------------|
| `path`     | clip filename, e.g. `TDH_recording_S001.wav`  |
| `sentence` | transcription, e.g. `gani cɵtcɵlo ham benthalni` |

Clips must be **mono WAV, 16 kHz, between 1 and 20 seconds**.

### Output

- Best model weights and processor files saved directly to `output_dir`
- `trainer_state.json` with the full loss/WER history for plotting

---

## 3. `prediction_wav2vec2.py` — Transcribe audio

For each WAV file in `--audio_dir_path`, this script:

1. Resamples the audio to 16 kHz
2. Cuts it into chunks based on silence
3. Predicts a transcription for each chunk
4. Writes predictions to `.tsv`, `.xml` (Pangloss), `.eaf` (ELAN), and `.txt`

```bash
python prediction_wav2vec2.py \
    --audio_dir_path /path/to/wav \
    --model_path     /path/to/model \
    --language       tdh \
    [--min_silence 500] [--silence_offset 10] [--no_words]
```

- `--min_silence` (ms): minimum silence duration used to split chunks
- `--silence_offset` (dB below dBFS): threshold offset for silence detection
- `--no_words`: omit the word tier from the `.eaf` output (word tier is
  included by default)

---

## Typical end-to-end workflow

```bash
# 1. Build the dataset
python make_dataset.py create_audio --wav_dir raw/audio --trans_dir raw/annotations --out_dir data --language tdh
python make_dataset.py create_dataset --path data/

# 2. Fine-tune
python fine_tune_xlsr_wav2vec2.py --train_tsv data/train.tsv --val_tsv data/valid.tsv --output_dir model --clips_dir data/clips

# 3. Transcribe new recordings
python prediction_wav2vec2.py --audio_dir_path new_recordings --model_path model --language tdh
```
