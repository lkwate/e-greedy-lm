from transformers import AutoTokenizer, DataCollatorWithPadding
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import datasets
from loguru import logger
from typing import List, Union, Dict
from functools import partial


def _collate_fn(features, data_collator: DataCollatorWithPadding):
    encoder_features = [
        {
            key[len("encoder_") :]: value
            for key, value in feat.items()
            if key.startswith("encoder_")
        }
        for feat in features
    ]
    decoder_features = [
        {
            key[len("decoder_") :]: value
            for key, value in feat.items()
            if key.startswith("decoder_")
        }
        for feat in features
    ]

    encoder_features = data_collator(encoder_features)
    decoder_features = data_collator(decoder_features)

    decoder_features["input_ids"] = torch.where(
        decoder_features["input_ids"] == data_collator.tokenizer.pad_token_id,
        -100,
        decoder_features["input_ids"],
    )
    batch = {
        **{"encoder_" + key: value for key, value in encoder_features.items()},
        **{"decoder_" + key: value for key, value in decoder_features.items()},
    }
    return batch


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
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
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
            "encoder_input_ids",
            "encoder_attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
        ]
        self.train = self._data_processing(self.train, "Training")
        self.validation = self._data_processing(self.validation, "Validation")
        self.test = self._data_processing(self.test, "Testing")

    def _data_processing(self, dataset: datasets.arrow_dataset.Dataset, name: str):
        logger.info(f"{name} data transformation...")
        dataset = dataset.map(self._transform)
        dataset.set_format(type="torch", columns=self.columns)
        logger.info(f"{name} data transformation completed.")
        return dataset

    def _transform(self, item):
        doc, summary = item["document"], item["summary"]
        doc_output = self.tokenizer(doc, truncation=True)
        summary_output = self.tokenizer(summary, truncation=True)

        output = {
            "encoder_input_ids": doc_output["input_ids"],
            "encoder_attention_mask": doc_output["attention_mask"],
            "decoder_input_ids": summary_output["input_ids"],
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
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.collate_fn = partial(_collate_fn, data_collator=self.data_collator)

    def _transform(self, item):
        context, question, answer = (
            item["context"],
            item["question"],
            item["answers"]["text"][0],
        )
        input_text = answer + self.tokenizer.cls_token + context
        context = self.tokenizer(input_text, truncation=True)
        question = self.tokenizer(question, truncation=True)

        output = {
            "encoder_input_ids": context["input_ids"],
            "encoder_attention_mask": context["attention_mask"],
            "decoder_input_ids": question["input_ids"],
            "decoder_attention_mask": question["attention_mask"],
        }
        return output

    def _data_processing(self, dataset: datasets.arrow_dataset.Dataset, name: str):
        logger.info(f"{name} data transformation...")
        dataset = dataset.map(self._transform)
        dataset.set_format(type="torch", columns=self.columns)
        logger.info(f"{name} data transformation completed.")
        return dataset

    def prepare_data(self) -> None:
        logger.info("Loading SQuAD dataset...")
        self.dataset = datasets.load_dataset("squad")

        self.train, self.validation = self.dataset["train"], self.dataset["validation"]
        self.columns = [
            "encoder_input_ids",
            "encoder_attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
        ]
        self.train = self._data_processing(self.train, "Training")
        self.validation = self._data_processing(self.validation, "Validation")

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
