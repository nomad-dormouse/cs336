#!/bin/bash

uv run python -m cs336_basics.train_transformer \
    --device cpu \
    --dataset TS \
    --vocab_size 10000 \
    --context_length 256 \
    --num_layers 8 \
    --d_model 128 \
    --num_heads 4 \
    --batch_size 64 \
    --val_batch_size 256 \
    --max_iters 1000 \
    --learning_rate 1e-3 \
    --min_learning_rate 1e-4 \
    --warmup_iters 50 \
    --cosine_cycle_iters 1000 \
    --weight_decay 0.1 \
    --beta1 0.9 \
    --beta2 0.95 \
    --grad_clip 1.0 \
    --eval_and_log_interval 10 \
    --checkpoint_interval 100

if [ $? -eq 0 ]; then
    echo "Script executed successfully!"
else
    echo "Script failed!"
    exit 1
fi
