import torch
from torch import optim
from torch.optim import optimizer
from transformers import AutoTokenizer
from utils import epsilon_greedy_transform_label, uid_variance_fn, OPTIMIZER_DIC
import torch.nn as nn
import pytorch_lightning as pl


class RLLMLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        action_table: torch.LongTensor,
        tokenizer: AutoTokenizer,
        learning_rate: float = 1e-5,
        k: int = 10,
        epsilon: int = 0.2,
        beta: int = 0.06,
        variance_type: str = "local",
        lr_factor: float = 0.1,
        lr_patience: int = 5,
        optimizer_name: str = "Adam",
    ):
        super(RLLMLightningModule, self).__init__()
        self.model = model
        self.epsilon = epsilon
        self.beta = beta
        self.action_table = action_table
        self.tokenizer = tokenizer
        self.k = k
        self.variance_type = variance_type
        self.learning_rate = learning_rate
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.optimizer_name = optimizer_name

    def configure_optimizers(self):
        optimizer = OPTIMIZER_DIC[self.optimizer_name](
            self.model.parameters(), lr=self.learning_rate
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=self.lr_factor, patience=self.lr_patience
        )
        output = {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }
        return output

    def _compute_loss(self, input_ids, attention_mask, labels):
        labels = epsilon_greedy_transform_label(
            labels, self.action_table, self.tokenizer, epsilon=self.epsilon
        )
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss, logits = output.loss, output.logits
        uid_variance = uid_variance_fn(logits, labels, variance_type=self.variance_type)

        output = {"likelihood": loss, "uid_variance": uid_variance}
        loss = loss + self.beta * uid_variance

        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        loss, output = self._compute_loss(input_ids, attention_mask, labels)
        output["train_loss"] = loss
        self.log_dict(output, prog_bar=True)

        return output

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        loss, output = self._compute_loss(input_ids, attention_mask, labels)
        output["val_loss"] = loss
        self.log_dict(output, prog_bar=True)

        return output
