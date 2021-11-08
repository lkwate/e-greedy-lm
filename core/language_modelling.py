import torch
from transformers import AutoTokenizer
from utils import epsilon_greedy_transform_label, uid_variance_fn
import torch.nn as nn


class RLLanguageModeling(nn.Module):
    def __init__(
        self,
        model,
        action_table: torch.LongTensor,
        tokenizer: AutoTokenizer,
        k: int = 10,
        epsilon: int = 0.1,
        beta: int = 0.06,
        variance_type: str = "local",
    ):
        super(RLLanguageModeling, self).__init__()
        self.model = model
        self.epsilon = epsilon
        self.beta = beta
        self.action_table = action_table
        self.tokenizer = tokenizer
        self.k = k
        self.variance_type = variance_type

    def forward(self, input_ids, attention_mask, labels):
        labels = epsilon_greedy_transform_label(
            labels, self.action_table, self.tokenizer
        )
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        base_loss, logits = output.loss, output.logits
        uid_variance = uid_variance_fn(logits, variance_type=self.variance_type)

        loss = base_loss + self.beta * uid_variance
        return loss
