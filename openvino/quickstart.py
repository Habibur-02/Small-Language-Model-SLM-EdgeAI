from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline

# ১. মডেল লোড এবং OpenVINO IR ফরম্যাটে কনভার্ট করা
model_id = "microsoft/DialoGPT-small"
print(f"Loading and converting {model_id}...")

ov_model = OVModelForCausalLM.from_pretrained(
    model_id, 
    export=True,
    compile=False
)

# ২. টোকেনাইজার লোড করা
tokenizer = AutoTokenizer.from_pretrained(model_id)

# ৩. কনভার্ট করা মডেলটি সেভ করা
save_directory = "models/dialogpt-openvino"
print(f"Saving model to {save_directory}...")
ov_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# ৪. ইনফারেন্সের জন্য লোড এবং কম্পাইল করা (CPU তে)
print("Loading optimized model for inference...")
ov_model = OVModelForCausalLM.from_pretrained(
    save_directory,
    device="CPU"  # ফাইল অনুযায়ী শুরুতে CPU দিয়ে টেস্ট করা নিরাপদ
)

# ৫. ইনফারেন্স পাইপলাইন তৈরি এবং রান করা
pipe = pipeline("text-generation", model=ov_model, tokenizer=tokenizer)
input_text = "Hello, how are you?"
result = pipe(input_text, max_length=50)

print("--- Result ---")
print(result)