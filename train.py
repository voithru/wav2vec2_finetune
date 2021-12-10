import warnings

import numpy as np
import yaml
from transformers import Trainer, TrainingArguments, Wav2Vec2ForCTC

from dataset import dataset
from utils import DataCollatorCTCWithPadding, compute_metrics


def train():
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    with open("config_train.yml") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    dataset_train, dataset_test, processor = dataset(args)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        args["pretrained_checkpoint_dir"],
        attention_dropout=args["attention_dropout"],
        hidden_dropout=args["hidden_dropout"],
        feat_proj_dropout=args["feat_proj_dropout"],
        mask_time_prob=args["mask_time_prob"],
        layerdrop=args["layerdrop"],
        # gradient_checkpointing=args["gradient_checkpointing"],
        ctc_loss_reduction=args["ctc_loss_reduction"],
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.freeze_feature_extractor()

    print("-------load_pretrained_model_done----------")

    training_args = TrainingArguments(
        output_dir=args["checkpoint_dir"],
        group_by_length=args["group_by_length"],
        per_device_train_batch_size=args["batch_size"],
        per_device_eval_batch_size=args["batch_size"],
        gradient_accumulation_steps=args["gradient_accumulation_steps"],
        evaluation_strategy=args["evaluation_strategy"],
        num_train_epochs=args["num_train_epochs"],
        fp16=args["fp16"],
        save_steps=args["save_steps"],
        eval_steps=args["eval_steps"],
        logging_steps=args["logging_steps"],
        weight_decay=args["weight_decay"],
        learning_rate=args["learning_rate"],
        warmup_steps=args["warmup_steps"],
        save_total_limit=args["save_total_limit"],
        dataloader_num_workers=args["dataloader_num_workers"],
    )

    print("-------train_ready_done---------")

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        tokenizer=processor.feature_extractor,
    )

    print("-------training_start!---------")
    trainer.train()
