import torch
import time
import json
import os
from datetime import datetime
from transformers import pipeline
import copy

# Initialize the DeepSeek pipeline
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,  # Enable nested quantization
            "bnb_4bit_quant_type": "nf4"       # Use Normal Float 4 data type
            },
    },
    # pad_token_id=128001,
    # batch_size=8,
    # temperature=0.6,
    device_map=0,
)

# Tokens limit for model's response
MAX_NEW_TOKENS = 256 * 4

# Generate a timestamped filename for saving the chat history
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
chat_history_dir = "chat_history"
chat_history_file = f"{chat_history_dir}/chat_history_{timestamp}.json"

# Create the chat history directory if it doesn't exist
os.makedirs(chat_history_dir, exist_ok=True)

# Initialize the chat history
chat_history = [
    {
        "role": "user",
        "content": "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user."
    }
    # {
    #     "role": "assistant",
    #     "content": "I am a knowledgeable, efficient, and direct AI assistant. I provide concise answers, focusing on the key information needed. I offer suggestions tactfully when appropriate to improve outcomes. I engage in productive collaboration with the user."
    # }
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
    
    t1 = time.time()
    # Generate the bot's response
    outputs = pipe(
        chat_history,
        temperature=0.6,  # recommended temperature 0.6 for deepseek R1
        batch_size=2,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    t2 = time.time()

    bot_response = outputs[0]["generated_text"][-1]['content']

    start_index = bot_response.find("<think>") + len("</think>")
    end_index = bot_response.find("</think>")

    bot_response_think = bot_response[start_index:end_index].strip("\n").strip()
    bot_response_content = bot_response[end_index + len("</think>"):].strip("\n").strip()

    print(f"Assistant ({t2 - t1:.2f}s): {bot_response_content}")
    
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
