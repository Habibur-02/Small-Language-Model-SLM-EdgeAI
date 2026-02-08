import nncf
from openvino import Core
from optimum.intel import OVModelForCausalLM
import torch
from transformers import AutoTokenizer

# Initialize NNCF for quantization
model_id = "microsoft/DialoGPT-small"

# Load model in OpenVINO format
ov_model = OVModelForCausalLM.from_pretrained(
    model_id, 
    export=True,
    compile=False
)

# Create calibration dataset for quantization
tokenizer = AutoTokenizer.from_pretrained(model_id)
calibration_data = [
    "Hello, how are you today?",
    "What is artificial intelligence?",
    "Tell me about machine learning.",
    "How does deep learning work?",
    "Explain neural networks."
]

def create_calibration_dataset():
    for text in calibration_data:
        tokens = tokenizer.encode(text, return_tensors="pt")
        yield {"input_ids": tokens}

# Apply post-training quantization
core = Core()
model = core.read_model(ov_model.model_path)

# Configure quantization
quantization_config = nncf.QuantizationConfig(
    input_info=nncf.InputInfo(
        sample_size=(1, 10),  # batch_size, sequence_length
        type="long"
    )
)

# Create quantized model
quantized_model = nncf.quantize_with_tune_runner(
    model,
    create_calibration_dataset(),
    quantization_config
)

# Save quantized model
import openvino as ov
ov.save_model(quantized_model, "models/dialogpt-quantized.xml")