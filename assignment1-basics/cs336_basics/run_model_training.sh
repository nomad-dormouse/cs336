#!/bin/bash
# 7 Experiments

# Problem (experiment_log): Experiment logging (3 points)


uv run python -m cs336_basics.train_transformer \
        --test_mode 0 \
        --device auto \

if [ $? -eq 0 ]; then
    echo "Script executed successfully!"
else
    echo "Script failed!"
    exit 1
fi
