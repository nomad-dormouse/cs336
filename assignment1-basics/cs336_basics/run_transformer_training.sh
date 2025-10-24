#!/bin/bash

uv run python -m cs336_basics.train_transformer \
    --train_data "results/tokenised/TinyStoriesV2-GPT4-train_tokenised.npy" \
    --val_data "results/tokenised/TinyStoriesV2-GPT4-valid_tokenised.npy" \
    --vocab_size 10000 \
    --context_length 64 \
    --num_layers 4 \
    --d_model 128 \
    --num_heads 4 \
    --batch_size 2 \
    --max_iters 10 \
    --learning_rate 1e-3 \
    --warmup_iters 5 \
    --cosine_cycle_iters 10 \
    --log_interval 2 \
    --eval_interval 5 \
    --checkpoint_interval 5 \
    --device cpu \
    --wandb_run_name tiny-stories-test

if [ $? -eq 0 ]; then
    echo "Script executed successfully!"
else
    echo "Script failed!"
    exit 1
fi
