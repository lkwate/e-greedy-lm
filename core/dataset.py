from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import datasets
from loguru import logger
from typing import List, Union, Dict


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

    def _collate_fn(self, features):
        if isinstance(features, list):
            features = [
                {key: value.tolist() for key, value in feature.items()}
                for feature in features
            ]
        elif isinstance(features, dict):
            features = {key: value.tolist() for key, value in features.items()}
        else:
            raise ValueError("features should be of type either list of dictionary")

        features = self.data_collator(features)

        if "labels" in features:
            features["labels"] = torch.where(
                features["labels"] == 1,
                self.data_collator.label_pad_token_id,
                features["labels"],
            )

        return features

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.validation, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )
