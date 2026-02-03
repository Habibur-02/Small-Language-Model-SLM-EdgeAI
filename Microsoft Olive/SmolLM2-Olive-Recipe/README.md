# SmolLM2-135M Optimization Recipe for CPU

This recipe optimizes the [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) model for efficient CPU inference using Microsoft Olive.

## Optimization Details
- **Precision:** INT4
- **Target Device:** CPU
- **Inference Engine:** ONNX Runtime GenAI

## How to Use

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

    Run optimization:
    Bash

    python -m olive run --config olive_config.json

    Run Inference: Use the generated model in models/smollm_manual with ONNX Runtime GenAI.