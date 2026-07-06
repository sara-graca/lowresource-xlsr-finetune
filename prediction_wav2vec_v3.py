# -*- coding: utf-8 -*-
"""
prediction_wav2vec.py — Transcribe audio with a fine-tuned Wav2Vec2 model.

For each wav file in --audio_dir_path:
  1. resample to 16 kHz
  2. cut into chunks on silence
  3. predict a transcription for each chunk
  4. write the predictions to .tsv, .xml (Pangloss), .eaf (ELAN) and .txt

USAGE
-----
    python prediction_wav2vec.py \
        --audio_dir_path /path/to/wav \
        --model_path     /path/to/model \
        --language       tdh \
        [--min_silence 500] [--silence_offset 10] [--no_words]

--min_silence (ms) and --silence_offset (dB below dBFS) control the silence chunking
--no_words omits the word tier from the .eaf (the word tier is on by default)
"""
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timezone

import torch
import librosa
import torchaudio
import pandas as pd
import soundfile as sf
import xml.etree.ElementTree as ET
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
from pydub.silence import split_on_silence

XML_NS = "http://www.w3.org/XML/1998/namespace"


def get_wav_files_from_directory(path, include_subdirs=True):
    if include_subdirs:
        return list(path.rglob("*.wav"))
    return list(path.glob("*.wav"))


def create_timestamps(df_results):
    df_results['Start'] = df_results['Duration'].cumsum() - df_results['Duration']
    df_results['End'] = df_results['Duration'].cumsum()
    return df_results


def predict_audio(model, processor, chunks_dir, corpus_name, tsv_filename):
    """Predict transcriptions for every chunk and write them to a TSV."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunks = sorted(get_wav_files_from_directory(chunks_dir, include_subdirs=False))
    predictions = []
    durations = []

    for chunk in chunks:
        speech_array, sampling_rate = torchaudio.load(str(chunk))
        speech = speech_array[0].numpy()
        speech = speech[:16_000 * 25]  # truncate to 25 seconds max
        durations.append(librosa.get_duration(y=speech, sr=sampling_rate))

        input_dict = processor(speech, return_tensors="pt", padding=True, sampling_rate=16_000)
        input_dict = input_dict.to(device)

        logits = model(input_dict.input_values.to(device)).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        prediction = processor.decode(pred_ids)
        predictions.append('' if pd.isna(prediction) else prediction)

    df_results = pd.DataFrame({'Prediction': predictions, 'Duration': durations, 'Chunk': chunks})
    df_results = create_timestamps(df_results)
    df_results = df_results[["Chunk", "Prediction", "Start", "End", "Duration"]]
    df_results.to_csv(tsv_filename, index=False)


def resample_audio(audio_file_path, resampled_file_path):
    signal, sampling_rate = librosa.load(audio_file_path)
    signal = librosa.resample(signal, orig_sr=sampling_rate, target_sr=16_000)
    sf.write(resampled_file_path, signal, 16000, 'PCM_16')
    return resampled_file_path


def cut_file(audio_file_path, chunks_dir, min_silence_len=500, silence_offset=10):
    """Cut an audio file into chunks on silence. Defaults (500 ms, 10 dB) are
    the values tuned for the Thulung corpus by grid search."""
    audio = AudioSegment.from_file(audio_file_path, format="wav")
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=audio.dBFS - silence_offset,
        keep_silence=True,
    )
    for i, chunk in enumerate(chunks):
        chunk_file = chunks_dir / f"chunk_{i:06d}.wav"
        chunk.export(chunk_file, format="wav")
    print(f"Cut into {len(chunks)} chunks.")


# ─── Output writers ───────────────────────────────────────────────────────────

def tsv2plaintxt(tsv_file_path, txt_file_path=None):
    """Write all predictions concatenated into a plain text file."""
    if not txt_file_path:
        txt_file_path = tsv_file_path.with_suffix(".txt")

    df_results = pd.read_csv(tsv_file_path)
    text = ""
    for _, row in df_results.iterrows():
        transcription = row['Prediction']
        if pd.isna(transcription) or transcription == '':
            continue
        text += " " + str(transcription)

    with open(txt_file_path, "w", encoding='utf-8') as fh:
        fh.write(text)
    print(f"Generating txt file : {txt_file_path}")


def tsv2xml(tsv_file_path, language, xml_file_path=None):
    """Write predictions to a Pangloss XML file, one <S> per chunk."""
    if not xml_file_path:
        xml_file_path = tsv_file_path.with_suffix(".xml")

    corpus_name = tsv_file_path.stem
    print(f"Generating xml file : {xml_file_path}")

    df_results = pd.read_csv(tsv_file_path)
    root = create_root(corpus_name, language)

    counter = 0
    for _, row in df_results.iterrows():
        transcription = row['Prediction']
        if pd.isna(transcription) or transcription == '':
            continue
        counter += 1
        sentence_id = corpus_name + "_S" + str(counter).zfill(3)
        s = add_sentence_element(root, sentence_id)
        add_audio_element(s, float(row['Start']), float(row['End']))
        add_form_element(s, str(transcription))
        add_word_elements(s, str(transcription))

    ET.ElementTree(root).write(xml_file_path, encoding='utf-8', xml_declaration=True)


def create_root(corpus_name, language):
    root = ET.Element("TEXT")
    root.set('id', corpus_name)
    root.set('xml:lang', language)
    return root


def add_sentence_element(root, sentence_id):
    s = ET.SubElement(root, "S")
    s.set('id', sentence_id)
    return s


def add_audio_element(s, start, end):
    audio = ET.SubElement(s, "AUDIO")
    audio.set('start', "{:.1f}".format(start))
    audio.set('end', "{:.1f}".format(end))


def add_form_element(s, transcription):
    form = ET.SubElement(s, "FORM")
    form.set('kindOf', 'phono')
    form.text = transcription


def add_word_elements(s, transcription):
    words = transcription.split(' ') if transcription else []
    for word in words:
        w = ET.SubElement(s, "W")
        form_word = ET.SubElement(w, "FORM")
        form_word.text = word


def _esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _esc_attr(text):
    return _esc(text).replace('"', "&quot;")


def tsv2eaf(tsv_file_path, language, media_file=None, eaf_file_path=None, words=True):
    """Write predictions to an ELAN .eaf file. The phono transcription goes on a
    time-aligned 'tx' tier, one annotation per chunk. If words=True, a 'word'
    tier (symbolic subdivision under tx) splits each chunk on spaces; the words
    carry no independent timing (ELAN distributes them within the chunk)."""
    if not eaf_file_path:
        eaf_file_path = tsv_file_path.with_suffix(".eaf")

    print(f"Generating eaf file : {eaf_file_path}")
    df_results = pd.read_csv(tsv_file_path)

    ann_id = 0
    ts_id = 0
    ts_slots = []          # (ts_id, ms)
    annotations = []       # (ann_id, ts1_id, ts2_id, value)
    word_annotations = []  # (ann_id, parent_ann_id, prev_ann_id_or_None, value)

    for _, row in df_results.iterrows():
        transcription = row['Prediction']
        if pd.isna(transcription) or transcription == '':
            continue
        ann_id += 1
        ts_id += 1
        ts1 = f"ts{ts_id}"
        ts_slots.append((ts1, round(float(row['Start']) * 1000)))
        ts_id += 1
        ts2 = f"ts{ts_id}"
        ts_slots.append((ts2, round(float(row['End']) * 1000)))
        phono_id = f"a{ann_id}"
        annotations.append((phono_id, ts1, ts2, str(transcription)))

        if words:
            prev = None
            for w in str(transcription).split():
                ann_id += 1
                wid = f"a{ann_id}"
                word_annotations.append((wid, phono_id, prev, w))
                prev = wid

    media = _esc_attr(media_file) if media_file else ""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<ANNOTATION_DOCUMENT AUTHOR="" DATE="{now}" FORMAT="3.0" VERSION="3.0"',
        '    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '    xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">',
        f'    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">',
    ]
    if media:
        lines.append(
            f'        <MEDIA_DESCRIPTOR MEDIA_URL="file://{media}"'
            f' MIME_TYPE="audio/x-wav"/>'
        )
    lines.extend([
        f'        <PROPERTY NAME="lastUsedAnnotationId">{ann_id}</PROPERTY>',
        '    </HEADER>',
        '    <TIME_ORDER>',
    ])
    for tsid, ms in ts_slots:
        lines.append(f'        <TIME_SLOT TIME_SLOT_ID="{tsid}" TIME_VALUE="{ms}"/>')
    lines.append('    </TIME_ORDER>')

    lang_attr = f' LANG_REF="{_esc_attr(language)}"' if language else ""
    lines.append(
        f'    <TIER{lang_attr} LINGUISTIC_TYPE_REF="default-lt" TIER_ID="tx">'
    )
    for aid, ts1, ts2, value in annotations:
        lines.extend([
            '        <ANNOTATION>',
            f'            <ALIGNABLE_ANNOTATION ANNOTATION_ID="{aid}"'
            f' TIME_SLOT_REF1="{ts1}" TIME_SLOT_REF2="{ts2}">',
            f'                <ANNOTATION_VALUE>{_esc(value)}</ANNOTATION_VALUE>',
            '            </ALIGNABLE_ANNOTATION>',
            '        </ANNOTATION>',
        ])
    lines.append('    </TIER>')

    if words and word_annotations:
        lines.append(
            '    <TIER LINGUISTIC_TYPE_REF="symsub" PARENT_REF="tx" TIER_ID="word">'
        )
        for aid, ref_id, prev_id, value in word_annotations:
            prev_attr = f' PREVIOUS_ANNOTATION="{prev_id}"' if prev_id else ""
            lines.extend([
                '        <ANNOTATION>',
                f'            <REF_ANNOTATION ANNOTATION_ID="{aid}"'
                f' ANNOTATION_REF="{ref_id}"{prev_attr}>',
                f'                <ANNOTATION_VALUE>{_esc(value)}</ANNOTATION_VALUE>',
                '            </REF_ANNOTATION>',
                '        </ANNOTATION>',
            ])
        lines.append('    </TIER>')

    lines.append(
        '    <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false"'
        ' LINGUISTIC_TYPE_ID="default-lt" TIME_ALIGNABLE="true"/>'
    )
    if words and word_annotations:
        lines.append(
            '    <LINGUISTIC_TYPE CONSTRAINTS="Symbolic_Subdivision"'
            ' GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="symsub"'
            ' TIME_ALIGNABLE="false"/>'
        )
    if language:
        lid = _esc_attr(language)
        lines.append(
            f'    <LANGUAGE LANG_DEF="{lid}" LANG_ID="{lid}" LANG_LABEL="{lid}"/>'
        )
    if words and word_annotations:
        lines.append(
            '    <CONSTRAINT DESCRIPTION="Symbolic subdivision of a parent'
            ' annotation. Annotations refering to the same parent are ordered"'
            ' STEREOTYPE="Symbolic_Subdivision"/>'
        )
    lines.append('</ANNOTATION_DOCUMENT>')

    with open(eaf_file_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process audio files for transcription.')
    parser.add_argument('--audio_dir_path', type=str, required=True,
                        help='Path to the directory with audio files')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--language', type=str, required=True,
                        help='ISO 639-3 language code, e.g. tdh')
    parser.add_argument('--min_silence', type=int, default=500,
                        help='Minimum silence length in ms for chunking (default: 500)')
    parser.add_argument('--silence_offset', type=int, default=10,
                        help='Silence threshold below dBFS in dB (default: 10)')
    parser.add_argument('--no_words', dest='words', action='store_false',
                        help='Do not add a word tier to the .eaf (on by default)')
    args = parser.parse_args()

    language = args.language
    model_path = Path(args.model_path)
    audio_dir_path = Path(args.audio_dir_path)

    audio_files = get_wav_files_from_directory(audio_dir_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    processor = Wav2Vec2Processor.from_pretrained(model_path)

    for audio_file_path in audio_files:
        corpus_name = audio_file_path.stem

        working_dir = audio_file_path.relative_to(audio_dir_path).parent
        working_dir = audio_dir_path.parent / (audio_dir_path.name + "_result") / working_dir
        working_dir.mkdir(exist_ok=True, parents=True)

        resampled_path = working_dir / Path(f"{corpus_name}_resampled.wav")
        resample_audio(audio_file_path, resampled_path)

        chunks_dir = working_dir / f"{corpus_name}_chunk"
        chunks_dir.mkdir(exist_ok=True)
        cut_file(resampled_path, chunks_dir,
                 min_silence_len=args.min_silence, silence_offset=args.silence_offset)
        resampled_path.unlink()

        tsv_file_path = working_dir / (corpus_name + ".tsv")
        predict_audio(model, processor, chunks_dir, corpus_name, tsv_file_path)

        shutil.rmtree(chunks_dir)

        tsv2xml(tsv_file_path, language)
        tsv2eaf(tsv_file_path, language, media_file=str(audio_file_path.resolve()),
                words=args.words)
        tsv2plaintxt(tsv_file_path)