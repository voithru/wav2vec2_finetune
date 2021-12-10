import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import librosa
import numpy as np
import soundfile
import torch
from datasets import load_metric
from transformers import Wav2Vec2Processor


def remove_special_characters(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:"\“\%\‘\”\�]'
    batch["transcript"] = (
        re.sub(chars_to_ignore_regex, "", batch["transcript"]).lower() + " "
    )
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["transcript"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def get_audio_from_path(file_path):
    standard_sample_rate = 16000
    if file_path.suffix == ".pcm":
        with file_path.open("rb") as audio_file:
            audio = np.fromfile(audio_file, dtype=np.int16).astype(np.single) / 32768
    else:
        audio, sample_rate = soundfile.read(file_path, dtype=np.single, always_2d=True)
        audio = audio.mean(axis=1)
        if sample_rate != standard_sample_rate:
            audio = librosa.resample(
                audio, sample_rate, standard_sample_rate, res_type="kaiser_fast"
            )

    return audio


def speech_file_to_array_fn(batch):
    audio = get_audio_from_path(Path(batch["audio_path"]))
    batch["speech"] = audio
    batch["sampling_rate"] = 16000
    batch["target_text"] = batch["transcript"]
    return batch


def compute_metrics(pred, processor):
    cer_metric = load_metric("cer")
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


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
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch
