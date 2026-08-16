# !/bin/bash

python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir "/path/to/llama_checkpoints/Llama3.1-8B/" \
    --model_size 8B \
    --llama_version 3.1 \
    --output_dir "/path/to/MasterThesis/utils/convert_llama_weights/meta_llama_model/Llama3.1-8B-hf/" \