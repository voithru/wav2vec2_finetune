import json
from datasets import load_dataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from utils import extract_all_chars, speech_file_to_array_fn

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


def dataset(args):
    all_dataset = load_dataset(
        "json",
        data_files={"train": args["train_data_path"], "test": args["test_data_path"]},
    )

    dataset_train = all_dataset["train"]
    dataset_test = all_dataset["test"]

    if args['make_vocab'] == True:
        make_vocab(dataset_train, dataset_test)
        print("------make_vocab_done------")

    dataset_train = dataset_train.map(
        speech_file_to_array_fn, remove_columns=dataset_train.column_names
    )
    dataset_test = dataset_test.map(
        speech_file_to_array_fn, remove_columns=dataset_test.column_names
    )

    print("------speech_file_to_array_done------")

    tokenizer = Wav2Vec2CTCTokenizer(
        args["vocab_path"],
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
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

    def prepare_dataset(batch):
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
        batch["input_values"] = processor(
            batch["speech"], sampling_rate=batch["sampling_rate"][0]
        ).input_values

        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
        return batch

    dataset_train = dataset_train.map(
        prepare_dataset,
        remove_columns=dataset_train.column_names,
        batch_size=4,
        num_proc=48,
        batched=True,
    )
    dataset_test = dataset_test.map(
        prepare_dataset,
        remove_columns=dataset_test.column_names,
        batch_size=4,
        num_proc=48,
        batched=True,
    )

    print("------prepare_dataset_done------")

    return dataset_train, dataset_test, processor