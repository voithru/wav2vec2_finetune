import numpy as np
import json
from datasets import load_metric

def compute_metrics(pred, processor):
    cer_metric = load_metric("cer")
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

def extract_all_chars(batch):
    all_text = " ".join(batch["transcript"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def speech_file_to_array_fn(batch):
    with open(batch["audio_path"], "rb") as audiofile:
        speech_array = np.fromfile(audiofile, dtype=np.int16).astype(np.single) / 32768
    sampling_rate = 16000
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["transcript"]
    return batch

def make_vocab(dataset_train, dataset_test):
    vocab_train = dataset_train.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset_train.column_names,
    )
    vocab_test = dataset_test.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset_test.column_names,
    )
    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open("vocab.json", "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)
