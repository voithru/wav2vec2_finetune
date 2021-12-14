import json
from pathlib import Path

import torch
import typer
import yaml
from tqdm import tqdm
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                          Wav2Vec2ForCTC, Wav2Vec2Processor)

from utils import get_audio_from_path, makedirs

app = typer.Typer()


@app.command(name="one")
def predict_one():
    with open("config_predict.yml") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    tokenizer = Wav2Vec2CTCTokenizer(
        args["vocab_path"],
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    model = Wav2Vec2ForCTC.from_pretrained(args["checkpoint_path"])
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    standard_sample_rate = 16000
    file_dir = Path(args["input_dir"])
    file_list = file_dir.glob("*.wav")
    for file_path in tqdm(file_list):
        audio = get_audio_from_path(file_path)

        inputs = processor(
            audio, sampling_rate=standard_sample_rate, return_tensors="pt"
        )
        logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = processor.batch_decode(predicted_ids)

        makedirs(args["output_dir"])
        file = open(
            args["output_dir"] + f"/{file_path.stem}.json", "w", encoding="utf-8"
        )
        line = json.dumps({"text": transcription}, ensure_ascii=False)
        file.write(line)
        file.close()


@app.command(name="all")
def predict_all():
    with open("config_predict.yml") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    tokenizer = Wav2Vec2CTCTokenizer(
        args["vocab_path"],
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    checkpoint_path = Path(args["checkpoint_dir"])
    checkpoints = checkpoint_path.glob("checkpoint-*")

    standard_sample_rate = 16000
    file_dir = Path(args["input_dir"])
    file_list = file_dir.glob("*.wav")
    audios = []
    for file_path in file_list:
        audio = get_audio_from_path(file_path)
        audios.append((file_path.stem, audio))

    makedirs(args["output_dir"])
    for checkpoint in tqdm(checkpoints):
        print(checkpoint.stem)
        model = Wav2Vec2ForCTC.from_pretrained(checkpoint)
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor, tokenizer=tokenizer
        )

        for (file_name, audio) in audios:
            inputs = processor(
                audio, sampling_rate=standard_sample_rate, return_tensors="pt"
            )
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            transcription = processor.batch_decode(predicted_ids)

            makedirs(args["output_dir"] + f"/{checkpoint.stem}")
            file = open(
                args["output_dir"] + f"/{checkpoint.stem}/{file_name}.json",
                "w",
                encoding="utf-8",
            )
            line = json.dumps({"text": transcription}, ensure_ascii=False)
            file.write(line)
            file.close()
