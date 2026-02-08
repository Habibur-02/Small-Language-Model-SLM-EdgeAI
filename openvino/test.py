from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline

# Load and convert model to OpenVINO IR format
model_id = "microsoft/DialoGPT-small"
ov_model = OVModelForCausalLM.from_pretrained(
    model_id, 
    export=True,
    compile=False
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save the converted model
save_directory = "models/dialogpt-openvino"
ov_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# Load and compile for inference
ov_model = OVModelForCausalLM.from_pretrained(
    save_directory,
    device="CPU"  # or "GPU", "AUTO"
)

# Create inference pipeline
pipe = pipeline("text-generation", model=ov_model, tokenizer=tokenizer)
result = pipe("Hello, how are you?", max_length=50)
print(result)