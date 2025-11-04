#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --root_path ./pridict_price_datasets/ \
  --model_id PriceForecast \
  --model TimesNet \
  --task_name forecast \
  --seq_len 24 \
  --pred_len 12 \
  --d_model 64 \
  --top_k 2 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 5 \
  --use_gpu 1