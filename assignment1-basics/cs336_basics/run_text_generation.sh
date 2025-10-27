#!/bin/bash

uv run python -m cs336_basics.generate_text \
    --model_filename "c256-n8-d128-h4-b64-i100-TS" \
    --prompt "Once upon a time" \
    --max_tokens 256 \
    --temperature 0.0 \
    --top_p 0.9

if [ $? -eq 0 ]; then
    echo "Script executed successfully!"
else
    echo "Script failed!"
    exit 1
fi
