import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import gc
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


gc.collect()
torch.cuda.set_device(0)
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

# Converting Bytes to Gigabytes
def b2gb(x):
    return x / 2**30

model_path = "/data/xiaxuan/models/Qwen2.5/3b-it"

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=1024)

tensor = torch.cuda.memory_allocated()
print(f"Before loading model, GPU Memory occupied by tensor: {b2gb(tensor):.2f} GB")
cache = torch.cuda.memory_reserved()
print(f"Before loading model, GPU Memory occupied by the caching allocator: {b2gb(cache):.2f} GB")

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model=model_path, dtype=torch.bfloat16, seed=42, gpu_memory_utilization=1.0, tensor_parallel_size=1, enforce_eager=False)

tensor = torch.cuda.memory_allocated()
print(f"After loading model, GPU Memory occupied by tensor: {b2gb(tensor):.2f} GB")
cache = torch.cuda.memory_reserved()
print(f"After loading model, GPU Memory occupied by the caching allocator: {b2gb(cache):.2f} GB")

# Prepare your prompts
# prompt = "Tell me something about large language models."
prompts = ["Tell me something about large language models.", "write a quick sort algorithm."]


def get_text(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text


texts = [get_text(prompt) for prompt in prompts]
# generate outputs
outputs = llm.generate(texts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\nPrompt: {prompt!r}\n\nGenerated text: {generated_text!r}")

tensor = torch.cuda.memory_allocated()
print(f"After generating outputs, GPU Memory occupied by tensor: {b2gb(tensor):.2f} GB")
cache = torch.cuda.memory_reserved()
print(f"After generating outputs, GPU Memory occupied by the caching allocator: {b2gb(cache):.2f} GB")
tensor_peak = torch.cuda.max_memory_allocated()
print(f"Peak GPU memory allocated: {b2gb(tensor_peak):.2f} GB")
cache_peak = torch.cuda.max_memory_reserved()
print(f"Peak GPU memory reserved: {b2gb(cache_peak):.2f} GB\n")
