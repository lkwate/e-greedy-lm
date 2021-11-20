from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import datasets
from loguru import logger
from typing import List, Union, Dict
from functools import partial


def _collate_fn(features, data_collator):
    if isinstance(features, list):
        features = [
            {key: value.tolist() for key, value in feature.items()}
            for feature in features
        ]
    elif isinstance(features, dict):
        features = {key: value.tolist() for key, value in features.items()}
    else:
        raise ValueError("features should be of type either list of dictionary")

    features = data_collator(features)

    if "labels" in features:
        features["labels"] = torch.where(
            features["labels"] == 1,
            data_collator.label_pad_token_id,
            features["labels"],
        )

    return features


class MultiNewsLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        batch_size: int,
        num_workers: int,
        max_length: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, max_length=self.max_length, padding="max_length"
        )
        self.collate_fn = partial(_collate_fn, data_collator=self.data_collator)

    def prepare_data(self):
        logger.info("Multi_news dataset loading....")
        self.dataset = datasets.load_dataset("multi_news")
        logger.info("Loading of multi_news datasets completed.")
        self.train, self.validation, self.test = (
            self.dataset["train"],
            self.dataset["validation"],
            self.dataset["test"],
        )
        self.columns = [
            "input_ids",
            "attention_mask",
            "decoder_attention_mask",
            "labels",
        ]

        logger.info("Training data transformation...")
        self.train = self.train.map(self._transform, batched=True)
        self.train.set_format(type="torch", columns=self.columns)
        logger.info("Training data transformation completed.")

        logger.info("Validation data transformation...")
        self.validation = self.validation.map(self._transform, batched=True)
        self.validation.set_format(type="torch", columns=self.columns)
        logger.info("Validation data transformation completed.")

        logger.info("Testing data transformation...")
        self.test = self.test.map(self._transform, batched=True)
        self.test.set_format(type="torch", columns=self.columns)
        logger.info("Testing data transformation completed.")

    def _transform(self, item):
        doc, summary = item["document"], item["summary"]
        doc_output = self.tokenizer(doc, truncation=True, padding="max_length")
        summary_output = self.tokenizer(summary, truncation=True, padding="max_length")

        output = {
            "input_ids": doc_output["input_ids"],
            "attention_mask": doc_output["attention_mask"],
            "labels": summary_output["input_ids"],
            "decoder_attention_mask": summary_output["attention_mask"],
        }

        return output

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.validation,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


class SQuADLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        batch_size: int,
        num_workers: int,
        max_length: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, max_length=self.max_length, padding="max_length"
        )
        self.collate_fn = partial(_collate_fn, data_collator=self.data_collator)

    def _transform(self, item):
        context, question, answers = (
            item["context"],
            item["question"],
            list(map(lambda x: x["text"][0], item["answers"])),
        )
        input_text = [
            a + self.tokenizer.cls_token + c for a, c in zip(answers, context)
        ]
        context = self.tokenizer(input_text, truncation=True, padding="max_length")
        question = self.tokenizer(question, truncation=False, padding="max_length")

        output = {
            "input_ids": context["input_ids"],
            "attention_mask": context["attention_mask"],
            "labels": question["input_ids"],
            "decoder_attention_mask": question["attention_mask"],
        }
        return output

    def prepare_data(self) -> None:

        logger.info("Loading SQuAD dataset...")
        self.dataset = datasets.load_dataset("squad")

        self.train, self.validation = self.dataset["train"], self.dataset["validation"]
        self.columns = [
            "input_ids",
            "attention_mask",
            "decoder_attention_mask",
            "labels",
        ]

        logger.info("Data transformation...")
        self.train = self.train.map(self._transform, batched=True)
        self.train.set_format(type="torch", columns=self.columns)

        self.validation = self.validation.map(self._transform, batched=True)
        self.validation.set_format(type="torch", columns=self.columns)
        logger.info("Data transformation completed.")

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.validation,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.val_dataloader()
