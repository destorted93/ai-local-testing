import torch
import time
import json
import os
from datetime import datetime
import scipy
import sounddevice as sd
from transformers import pipeline, AutoProcessor, BarkModel

# Initialize the Qwen pipeline
model_id = "Qwen/Qwen2.5-7B-Instruct"

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
    device_map=0,
)

# Tokens limit for model's response
MAX_NEW_TOKENS = 2000

# Generate a timestamped filename for saving the chat history
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
chat_history_dir = "chat_history"
chat_history_file = f"{chat_history_dir}/chat_history_{timestamp}.json"

# Create the chat history directory if it doesn't exist
os.makedirs(chat_history_dir, exist_ok=True)

# Initialize the chat history
chat_history = [
    {
        "role": "system",
        "content": "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user."
    }
]



# Function to save the chat history to a file
def save_chat_history():
    with open(chat_history_file, "w") as file:
        json.dump(chat_history, file, indent=4)

# Function to get a response
def chat_with_bot(user_input):
    # Append the user message to the chat history
    chat_history.append({"role": "user", "content": user_input})
    
    # Generate the bot's response
    outputs = pipe(
        chat_history,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    bot_response = outputs[0]["generated_text"][-1]['content']
    
    # Add the bot's response to the chat history
    chat_history.append({"role": "assistant", "content": bot_response})
    
    # Save the updated chat history
    save_chat_history()
    
    return bot_response

# Example usage
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye, friend!")
        break

    t1 = time.time()
    bot_response = chat_with_bot(user_input)
    t2 = time.time()

    print(f"Assistent ({t2 - t1:.2f}s): {bot_response}")
