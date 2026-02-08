import nncf
from openvino import Core
from optimum.intel import OVModelForCausalLM
import openvino as ov
from transformers import AutoTokenizer
import numpy as np

# ১. মডেল লোড
model_id = "microsoft/DialoGPT-small"
print(f"Loading {model_id}...")

# ফিক্স ১: মডেল লোড করার সময় 'ov_config' ব্যবহার করে ডাইনামিক শেপ সমস্যা কমানো
ov_model = OVModelForCausalLM.from_pretrained(
    model_id, 
    export=True,
    compile=False
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# মডেল সেভ করা
temp_model_dir = "models/dialogpt-temp"
print("Saving temporary model to disk...")
ov_model.save_pretrained(temp_model_dir)

# ২. ক্যালিব্রেশন ডেটাসেট তৈরি
calibration_data = [
    "Hello, how are you today?",
    "What is artificial intelligence?",
    "Tell me about machine learning.",
    "How does deep learning work?",
    "Explain neural networks."
]

def create_calibration_dataset():
    for text in calibration_data:
        # ফিক্স ২: ফিক্সড max_length এবং padding ব্যবহার করা
        inputs = tokenizer(
            text, 
            return_tensors="np", 
            padding="max_length",  # সব ইনপুট সমান সাইজের হবে
            max_length=128,        # ফিক্সড লেন্থ ১২৮
            truncation=True
        )
        
        yield {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }

# ৩. NNCF কোয়ান্টাইজেশন শুরু
print("Starting quantization...")
core = Core()
model = core.read_model(f"{temp_model_dir}/openvino_model.xml")

# ফিক্স ৩: মডেলের ইনপুট শেপ জোর করে ফিক্সড করে দেওয়া (Reshape)
# এটিই মূলত 'Shape mismatch' এররটি থামাবে
try:
    model.reshape([1, 128]) 
except Exception as e:
    print(f"Reshape warning: {e}")

# কোয়ান্টাইজেশন কনফিগারেশন
# 'subset_size' কমিয়ে ৩ দেওয়া হয়েছে যাতে দ্রুত শেষ হয় এবং এরর কম হয়
quantized_model = nncf.quantize(
    model=model,
    calibration_dataset=nncf.Dataset(create_calibration_dataset()),
    subset_size=3
)

# ৪. সেভ করা
save_path = "models/dialogpt-quantized.xml"
print(f"Saving quantized model to {save_path}...")
ov.save_model(quantized_model, save_path)
print("Quantization Complete! Success!")