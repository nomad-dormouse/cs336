#!/bin/bash
# 7 Experiments

# Problem (generate): Generate text (1 point)


uv run python -m cs336_basics.generate_text \
    --device cpu \
    --model_filename "v10000-c256-d512-f1344-l4-h16-b32-r0.001-i5000-TS-cpu" \
    --prompt "Once upon a time" \
    --max_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9

if [ $? -eq 0 ]; then
    echo "Script executed successfully!"
else
    echo "Script failed!"
    exit 1
fi
