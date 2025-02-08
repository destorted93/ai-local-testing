from transformers import  BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import time

model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Enable nested quantization
    bnb_4bit_quant_type="nf4",       # Use Normal Float 4 data type
    bnb_4bit_compute_dtype=torch.bfloat16,
    # llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype="auto",
    device_map=0
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# System role
SYSTEM_ROLE = "You are an expert AI assistant."

# Example user queries
prompts = [
    "1 + 0 = ?",
    "1 + 1 = ?",
    "1 + 2 = ?",
    "1 + 3 = ?",
    "1 + 4 = ?",
    "1 + 5 = ?",
    "1 + 6 = ?",
    "1 + 7 = ?",
    "1 + 8 = ?",
    "1 + 9 = ?",
    "1 + 10 = ?",
    "1 + 11 = ?",
    "1 + 12 = ?",
    "1 + 13 = ?",
    "1 + 14 = ?",
    "Tell me more about async in javascript, with examples",
    "1 + 15 = ?",
    "1 + 16 = ?",
    "1 + 17 = ?",
    "1 + 18 = ?",
    "1 + 19 = ?",
    "1 + 20 = ?",
    "1 + 21 = ?",
    "1 + 22 = ?",
    "1 + 23 = ?",
    "1 + 24 = ?",
    "1 + 25 = ?",
    "1 + 26 = ?",
    "1 + 27 = ?",
    "1 + 28 = ?",
    "1 + 29 = ?",
]

# Tokenize the input batch
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

t1 = time.time()
# Generate responses
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512)
t2 = time.time()

# Decode the responses
responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Print results
for i, response in enumerate(responses):
    print(i, "####################################################################")
    print(f"Response {i+1}:\n{response}\n")

print("Reponse in ", t2 - t1, "s")