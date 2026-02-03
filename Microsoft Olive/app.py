import gradio as gr
import onnxruntime_genai as og

model_path = 'models/qwen3-cpu-int8'

print("Loading model... please wait.")
try:
    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def chat_response(message, history):
    prompt = ""
    
    for turn in history:
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            user_msg = turn[0]
            bot_msg = turn[1]
            prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{bot_msg}<|im_end|>\n"

    prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"

    input_tokens = tokenizer.encode(prompt)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=512)
    params.input_ids = input_tokens

    generator = og.Generator(model, params)
    
    generated_text = ""
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()
        
        new_token = generator.get_next_tokens()[0]
        word = tokenizer.decode([new_token])
        generated_text += word
        yield generated_text

demo = gr.ChatInterface(
    fn=chat_response,
    title="🤖 My AI Chatbot (Qwen 0.5B ONNX)",
    description="Running locally on CPU with Microsoft Olive Optimization!"
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)