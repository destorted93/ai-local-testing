import torch
import time
import json
import os
from datetime import datetime
import copy
from transformers import  BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Initialize the DeepSeek pipeline
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Enable nested quantization
    bnb_4bit_quant_type="nf4",       # Use Normal Float 4 data type
    bnb_4bit_compute_dtype=torch.float16,
    # llm_int8_enable_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    # torch_dtype=torch.int8,
    quantization_config=bnb_config,
    device_map=0,
)

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True )

# Tokens limit for model's response
MAX_NEW_TOKENS = 4000

# Generate a timestamped filename for saving the chat history
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
chat_history_dir = "chat_history"
chat_history_file = f"{chat_history_dir}/chat_history_{timestamp}.json"

# Create the chat history directory if it doesn't exist
os.makedirs(chat_history_dir, exist_ok=True)

# Initialize the chat history
chat_history = [
    # {
    #     "role": "user",
    #     "content": "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user."
    # }
    {
        "role": "user",
        "content": "You are a knowledgeable, efficient, and direct AI assistant. You provide concise answers, focusing on the key information needed. You offer suggestions tactfully when appropriate to improve outcomes. You engage in productive collaboration with the user."
    }
    # {
    #     "role": "user",
    #     "content": ("You are an AI assistant."
    #                 "You are from Romania."
    #                 "Your name is Andrei."
    #                 "You are 22 years old."
    #                 "You speak only English and Romanian."
    #                 ),
    # },
    # {
    #     "role": "user",
    #     "content": "Your are a pirate which answers in pirate slang!"
    # }
]

# load system role content defined in external file and overwrite the default one
# with open("system_role.txt", 'r', encoding="utf-8") as f:
#     chat_history[0]["content"] = f.read()

copied_chat_history = copy.deepcopy(chat_history)



# Function to save the chat history to a file
def save_chat_history():
    with open(chat_history_file, "w") as file:
        json.dump(copied_chat_history, file, indent=4)

# Function to get a response
def chat_with_bot(user_input):
    # Append the user message to the chat history and copied chat history
    chat_history.append({"role": "user", "content": user_input})
    copied_chat_history.append({"role": "user", "content": user_input})

    inputs = tokenizer.apply_chat_template(
      chat_history,
      tokenize=True,
      add_generation_prompt=True,
      return_tensors="pt",
      return_dict=True,
    ).to(model.device)

    generation_kwargs = dict(**inputs, 
        do_sample=True,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.6,
        streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
    outputs = ""

    t1 = time.time()
    thread.start()

    for new_text in streamer:
        outputs += new_text
        print(new_text, end='', flush=True)

    t2 = time.time()

    # bot_response = outputs[0]["generated_text"][-1]['content']
    bot_response = outputs

    start_index = bot_response.find("<think>") + len("</think>")
    end_index = bot_response.find("</think>")

    bot_response_think = bot_response[start_index:end_index].strip("\n").strip()
    bot_response_content = bot_response[end_index + len("</think>"):].strip("\n").strip()

    print(f"\nAssistant ({t2 - t1:.2f}s): {bot_response_content}")
    
    # Add the bot's response to the chat history and copied chat history
    chat_history.append({"role": "assistant", "content": bot_response_content})
    copied_chat_history.append({"role": "assistant", "think": bot_response_think, "content": bot_response_content, "time": t2 - t1})
    
    # Save the updated chat history
    save_chat_history()
    
    return bot_response_content

# Example usage
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye, friend!")
        break

    bot_response = chat_with_bot(user_input)
