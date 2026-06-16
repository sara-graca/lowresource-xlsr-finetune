"""
USAGE
-----
    python fine_tune_xlsr_wav2vec2.py \
        --train_tsv  /path/to/train.tsv \
        --val_tsv   /path/to/val.tsv \
        --output_dir /path/to/output \
        --clips_dir  /path/to/clips \
        --base_model /path/to/other/model   (optional, defaults to facebook/wav2vec2-large-xlsr-53)

If output_dir already contains checkpoints, training resumes automatically.

INPUT
-----
TSV files: two tab-separated columns, no header.
    path        clip filename, e.g. TDH_recording_S001.wav
    sentence    transcription, e.g. gani cɵtcɵlo ham benthalni
Clips must be mono WAV at 16 kHz, between 1 and 20 seconds.

OUTPUT
------
Best model weights and processor files saved directly to output_dir.
trainer_state.json contains the full loss/WER history for plotting.
"""

import inspect
import json
import time

from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import torch
import torchaudio
import shutil
import argparse

from datasets import load_dataset
import evaluate
from dataclasses import dataclass
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

SAMPLING_RATE = 16_000


def extract_all_chars(batch):
    all_text = "|".join(batch["sentence"])
    all_text = all_text.replace(" ", "|")
    return {"vocab": list(set(all_text))}


def wav2array(batch):
    speech_array, sampling_rate = torchaudio.load(str(CLIPS_DIR / batch["path"]))
    assert (
        speech_array.shape[0] == 1
    ), f"{batch['path']} is stereo file --- only mono files can be considered"
    assert (
        sampling_rate == SAMPLING_RATE
    ), f"The sampling rate of your data must be {SAMPLING_RATE:,}. {batch['path']} has a sampling rate of {sampling_rate:,}"
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch


def prepare_dataset(batch):
    assert all(
        sampling_rate == SAMPLING_RATE for sampling_rate in batch["sampling_rate"]
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(
        batch["speech"], sampling_rate=batch["sampling_rate"][0]
    ).input_values

    batch["labels"] = processor.tokenizer(batch["target_text"]).input_ids

    return batch


def preprocess_split(dataset):
    """Load audio into arrays then tokenize. Same steps for train and val."""
    dataset = dataset.map(wav2array, remove_columns=dataset.column_names)
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names,
        batch_size=8,
        num_proc=4,
        batched=True,
    )
    return dataset


@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": float(cer), "wer": float(wer)}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tsv", type=str, required=True)
    parser.add_argument("--val_tsv",  type=str, required=True)
    parser.add_argument("--output_dir", type=lambda x: Path(x), required=True)
    parser.add_argument("--clips_dir", type=Path, required=True)
    parser.add_argument("--base_model", type=str, default="facebook/wav2vec2-large-xlsr-53")
    args = parser.parse_args()

    CLIPS_DIR = args.clips_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_dataset("csv", data_files=[args.train_tsv], delimiter="\t")["train"]
    val_data  = load_dataset("csv", data_files=[args.val_tsv],  delimiter="\t")["train"]

    vocab_train = train_data.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=train_data.column_names,
    )
    vocab_val = val_data.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=val_data.column_names,
    )

    vocab = sorted(set(vocab_train["vocab"]) | set(vocab_val["vocab"]))
    vocab = {v: k for k, v in enumerate(vocab)}
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    with open(args.output_dir / "vocab.json", "w") as vocab_file:
        json.dump(vocab, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer(
        args.output_dir / "vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLING_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    processor.save_pretrained(args.output_dir)

    # Reload so special tokens added on save are included in vocab size count.
    processor = Wav2Vec2Processor.from_pretrained(args.output_dir)
    print(f"Processor vocab size after reload: {len(processor.tokenizer)}")

    train_data = preprocess_split(train_data)
    val_data  = preprocess_split(val_data)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        ignore_mismatched_sizes=True,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.075,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ctc_zero_infinity=True,
    ).to("cuda")

    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable({"use_reentrant": False})

    # Verify trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}  ({100*trainable/total:.1f}%)")
    assert trainable > 0, "No trainable parameters!"

    # Pre-training sanity check: catch broken weight loading (nan logits)
    model.eval()
    with torch.no_grad():
        sample = train_data[0]
        input_vals = torch.tensor(sample["input_values"]).unsqueeze(0).to("cuda")
        logits = model(input_vals).logits
        assert not torch.isnan(logits).any(), "nan in logits before training — check weight loading"
    model.train()

    eval_strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        **{eval_strategy_key: "steps"},
        num_train_epochs=60,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=50,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=15,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer_tok_key = (
        "processing_class"
        if "processing_class" in inspect.signature(Trainer.__init__).parameters
        else "tokenizer"
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
        **{trainer_tok_key: processor.feature_extractor},
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    existing = sorted(
        args.output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    resume = existing[-1] if existing else None
    if resume:
        print(f"Resuming from {resume.name}")

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    # keep only best checkpoint, move to output dir root
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:
        best_ckpt = Path(best_ckpt)
        # copy best checkpoint contents up to the output dir root
        for f in best_ckpt.iterdir():
            shutil.copy2(f, args.output_dir / f.name)
        # delete all checkpoint-xxx folders
        for ckpt in args.output_dir.glob("checkpoint-*"):
            shutil.rmtree(ckpt)
        print(f"Best checkpoint ({best_ckpt.name})")
    elapsed = time.time() - t0
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f"\nTraining finished in {h}h {m}m {s}s")
    print(f"Best WER: {trainer.state.best_metric:.4f} at step {trainer.state.best_model_checkpoint}")
