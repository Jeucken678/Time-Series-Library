#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python run.py \
--is_training 1 \
--root_path "D:/Git-space/Times-Series-Net/Time-Series-Library/price/price_imputation_datasets"
--model_id TSL_Imputation \
--model_id PriceImputation \
--model TimesNet \
--data real_estate \
--task_name imputation \
--model TimesNet \
 --input_size 1 \
--task_name imputation \
--seq_len 24 \
--pred_len 24 \
--d_model 64 \
--top_k 2 \
--batch_size 32 \
--learning_rate 5e-5 \
--train_epochs 50 \
--patience 5 \
--use_gpu 1 \
--data price_imputation \
--input_size 1