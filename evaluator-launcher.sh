#!/bin/bash

model_name="roberta-base"
action_table_file="core/local_actions/local_action_index.csv"
batch_size=32
#strategy="ddp_spawn"
strategy="ddp"
### Question Generation
dataset_name="squad"
### Text Summarization
#dataset_name="multi_news"

#log_dir="../log_files"
log_dir="log_files"
checkpoint_path=${log_dir}/epoch=6-val_loss=3.8788.ckpt
output_file=${log_dir}/"eval.txt"
split="eval"
limit_batches=-1

python3 -m core.evaluator \
		$model_name \
		$action_table_file \
		$dataset_name \
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
		--optimizer_name Adam \
		--checkpoint_path $checkpoint_path \
		--output_file $output_file \
		--split $split \
		--limit_batches $limit_batches
