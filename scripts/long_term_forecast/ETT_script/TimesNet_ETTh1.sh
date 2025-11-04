#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model TimesNet \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_model 64 \
  --top_k 2 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --patience 3 \
  --des 'Exp_h1_96_96' \
  --task_name long_term_forecast \
  --inverse 0 \
  --use_gpu 1 \
  --gpu_id 0 \
  --use_multi_gpu 0