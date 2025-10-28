#!/bin/bash
# 7 Experiments

# Problem (experiment_log): Experiment logging (3 points)


uv run python -m cs336_basics.train_transformer \
        --test_mode 0 \
        --device auto \
        --dataset TS \
        --vocab_size 10000 \
        --context_length 256 \
        --d_model 512 \
        --d_ff 1344 \
        --num_layers 4 \
        --num_heads 16 \
        --batch_size 32 \
        --val_batch_size 256 \
        --max_iters 100 \
        --warmup_iters 10 \
        --cosine_cycle_iters 100 \
        --learning_rate 1e-3 \
        --min_learning_rate 1e-4 \
        --beta1 0.9 \
        --beta2 0.95 \
        --grad_clip 1.0 \
        --weight_decay 0.1 \
        --eval_and_log_interval 5 \
        --checkpoint_interval 20 \
        --wandb_project "cs336-assignment1" \

if [ $? -eq 0 ]; then
    echo "Script executed successfully!"
else
    echo "Script failed!"
    exit 1
fi
