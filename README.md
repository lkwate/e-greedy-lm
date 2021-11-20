# KABROLG
K-armed Bandits Reward-based Optimization for Language Generation

ϵ-greedy policy for language modelling regularized with the minimization of the uniform information density of the utterances generated [A Cognitive Regularizer for Language Modeling](https://arxiv.org/abs/2105.07144)

## Action Table Building
Construction of the table of actions using the embedding layer of Roberta-Base
```sh
python3 core/action.py roberta-base core/local_actions/local_action_index.csv --name=index
python3 core/action.py roberta-base core/local_actions/local_action_name.csv --name=name
```

## Training 
Training the LM on question generation (using the dataset **SQuAD**) or text summarization (using the dataset **Multi News**)
```sh
bash trainer-launcher.sh
```

## Generate sentences
```sh
bash evaluator-launcher.sh
```
