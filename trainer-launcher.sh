#!/bin/bash

model_name="roberta-base"
action_table_file="core/local_actions/local_action_index.csv"
#log_dir="../log_files"
log_dir="log_files"
batch_size=32
#strategy="ddp_spawn"
strategy="ddp"
### Question Generation
dataset_name="squad"
### Text Summarization
#dataset_name="multi_news"

python3 -m core.trainer \
		$model_name \
		$action_table_file \
		$dataset_name \
		$log_dir \
		--batch_size $batch_size \
		--num_workers 4 \
		--max_length 512 \
		--learning_rate 0.00001 \
		--k 10 \
		--epsilon 0.2 \
		--beta 0.06 \
		--variance_type local \
		--lr_factor 0.1 \
		--lr_patience 4 \
		--early_stopping_patience 5 \
		--optimizer_name Adam \
		--max_epochs 10 \
		--val_check_interval 0.25 \
		--accumulate_grad_batches 1 \
		--save_top_k 5 \
		--strategy $strategy \
		--random_seed 2021 \
		--add_variance