from transformers import (
    AutoTokenizer,
    AutoModel,
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
    EncoderDecoderModel,
)
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

OPTIMIZER_DIC = {"Adam": optim.Adam}


def create_action(
    embedding: torch.Tensor,
    tokenizer: AutoTokenizer,
    output_file: str,
    k: int = 10,
    factor: int = 10,
    name: bool = True,
):
    out = []
    topk = k * factor
    for _, index in tqdm(tokenizer.get_vocab().items()):
        corr = torch.matmul(
            embedding,
            embedding[index, :].reshape(
                -1,
                1,
            ),
        ).reshape(-1)
        related_tokens = corr.topk(topk)[-1].tolist()
        related_tokens = {
            tokenizer.decode(token).lower().strip(): 1 for token in related_tokens
        }
        if name:
            related_tokens = [
                token for token in list(related_tokens.keys())[:k] if token != ""
            ]
        else:
            related_tokens = [
                tokenizer.encode(token)[-2]
                for token in list(related_tokens.keys())[:k]
                if token != ""
            ]
        out.append((str(related_tokens)))

    out = pd.DataFrame(data=out, columns=["close tokens"])
    out.to_csv(output_file, index=False)


def epsilon_greedy_transform_label(
    labels: torch.LongTensor,
    action: torch.LongTensor,
    tokenizer: AutoTokenizer,
    k=10,
    epsilon: float = 0.1,
):
    transformed_labels = labels.clone()
    probability_matrix = torch.full(labels.shape, epsilon).to(labels.device)
    probability_matrix = torch.bernoulli(probability_matrix).bool()

    special_token_mask = (
        (labels != tokenizer.cls_token_id)
        & (labels != tokenizer.sep_token_id)
        & (labels != -100)
    )
    mask = probability_matrix & special_token_mask
    replace_indices = mask.nonzero(as_tuple=True)
    replace_token_indices = labels[replace_indices]
    next_action_indices = (
        torch.randint(0, k, replace_token_indices.shape).to(labels.device).long()
    )
    next_action_token_indices = action[replace_token_indices, next_action_indices]
    transformed_labels[replace_indices] = next_action_token_indices.to(labels.device)

    return transformed_labels


def uid_variance_fn(
    logits: torch.FloatTensor, labels: torch.LongTensor, variance_type: int = "local"
) -> torch.FloatTensor:
    uid = labels.clone()
    mask = uid != -100
    label_index = mask.nonzero(as_tuple=True)
    logits_index = label_index + (labels[label_index],)
    uid = torch.where(uid == -100, 0, uid).type_as(logits)

    uid[label_index] = logits[logits_index]
    scale = mask.sum(dim=-1)

    if variance_type == "local":
        var = uid[:, :-1] - uid[:, 1:]
        var = var ** 2
        var = var.sum(dim=-1) / scale
    else:
        means = uid.sum(dim=-1) / scale
        means = means.unsqueeze(-1)
        var = uid - means
        var = var ** 2
        var = var * mask
        var = var.sum(dim=-1) / scale

    var = var.mean()
    return var


def action_table_from_file(input_file: str, k: int) -> torch.LongTensor:
    actions = pd.read_csv(input_file)
    table = torch.zeros((len(actions), k)).long()

    for idx in tqdm(range(len(actions))):
        row = actions.iloc[idx]
        row = eval(row["close tokens"])
        idx = row[0]
        if len(row) > k:
            row = row[:k]
        elif len(row) < k:
            row = [row[0]] * (k - len(row)) + row
        table[idx] = torch.LongTensor(row)
    return table


def build_model(model_name, full_model):
    model, tokenizer = None, None
    if full_model:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer.cls_token = tokenizer.eos_token
        tokenizer.sep_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name)
        decoder = AutoModelForCausalLM.from_pretrained(
            model_name, is_decoder=True, add_cross_attention=True
        )
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size

    return model, tokenizer
