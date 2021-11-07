from transformers import AutoTokenizer
import pandas as pd
import torch
from tqdm import tqdm

def create_action(embedding: torch.Tensor, tokenizer: AutoTokenizer, output_file:str, k: int = 10, factor: int = 10, name: bool = True):
    output = []
    topk = k * factor
    for idx in tqdm(range(embedding.shape[0])):
        corr = torch.matmul(embedding, embedding[idx, :].reshape(-1, 1)).reshape(-1)
        related_tokens = corr.topk(topk)[-1].tolist()
        related_tokens = {tokenizer.decode(token).lower().strip(): True for token in related_tokens}
        
        if name:
            related_tokens = [token for token in list(related_tokens.keys())[:10]]
        else:
            related_tokens = [tokenizer.encode(token)[1] for token in list(related_tokens.keys())[:10]]
        token = tokenizer.decode(idx)
        output.append((idx, token, related_tokens))
    
    output = pd.DataFrame(data=output, columns=["index", "token", "close tokens"])
    output.to_csv(output_file, index=False)
    

def epsilon_greedy_transform_label(labels: torch.LongTensor, action: torch.Tensor, tokenizer: AutoTokenizer, k = 10, epsilon: float = 0.1):
    transformed_labels = labels.clone()
    probability_matrix = torch.full(labels.shape, epsilon)
    probability_matrix = torch.bernoulli(probability_matrix).bool()
    
    special_token_mask = (labels != tokenizer.cls_token_id) | (labels != tokenizer.sep_token_id) | (labels != -100)
    mask = probability_matrix & special_token_mask

    replace_indices = mask.nonzero(as_tuple=True)
    replace_token_indices = labels[replace_indices]
    
    next_action_indices = torch.randint(0, k, replace_token_indices.shape)
    next_action_token_indices = action[replace_token_indices, next_action_indices]
    transformed_labels[replace_indices] = next_action_token_indices

    return transformed_labels