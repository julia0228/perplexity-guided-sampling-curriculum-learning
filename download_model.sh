#!/usr/bin/env bash
# Download and cache the default model without starting training.
# Usage: bash download_model.sh

set -euo pipefail

MODEL_ID="Qwen/Qwen2.5-Math-1.5B"
MIRROR="${HF_ENDPOINT:-}"

if [ -n "$MIRROR" ]; then
    echo "Using mirror: $MIRROR"
else
    echo "Using default HuggingFace endpoint (set HF_ENDPOINT for mirror)"
fi

echo ">>> Downloading model: ${MODEL_ID}"
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_id = '${MODEL_ID}'
print(f'Downloading config...')
AutoConfig.from_pretrained(model_id)
print(f'Downloading tokenizer...')
AutoTokenizer.from_pretrained(model_id)
print(f'Downloading model weights (this may take a while)...')
AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto')
print('Done! Model cached in ~/.cache/huggingface/hub/')
"

echo ">>> Download complete: ${MODEL_ID}"
