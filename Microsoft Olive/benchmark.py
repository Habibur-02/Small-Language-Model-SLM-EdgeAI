import onnxruntime_genai as og
import time

model_path = 'models/qwen3-gpu-int8'

print(f"Loading model from {model_path}...")
model = og.Model(model_path)
tokenizer = og.Tokenizer(model)

prompt = "Explain quantum computing in simple terms."
chat_template = '<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n'
input_tokens = tokenizer.encode(chat_template.format(input=prompt))

params = og.GeneratorParams(model)
params.set_search_options(max_length=200)
params.input_ids = input_tokens

print("Generating response...")
start_time = time.time()

generator = og.Generator(model, params)
token_count = 0

while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()
    token_count += 1

end_time = time.time()
total_time = end_time - start_time
tps = token_count / total_time

print(f"\n--- Result for {model_path} ---")
print(f"Total Tokens: {token_count}")
print(f"Total Time: {total_time:.4f} seconds")
print(f"Speed: {tps:.2f} tokens/second")