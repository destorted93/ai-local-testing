import torch
import time
import json
import os
from datetime import datetime
import copy
from transformers import  BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Initialize the Qwen model
# model_id = "Qwen/Qwen2.5-14B-Instruct"
# model_id = "Qwen/Qwen2.5-7B-Instruct"
# model_id = "Qwen/Qwen2.5-3B-Instruct"
# model_id = "Qwen/Qwen2.5-1.5B-Instruct"
# model_id = "Qwen/Qwen2.5-0.5B-Instruct"
# model_id = "Qwen/Qwen2.5-Coder-14B-Instruct"
model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
# model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
# model_id = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
# model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

device = torch.device("cpu")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Enable nested quantization
    bnb_4bit_quant_type="nf4",       # Use Normal Float 4 data type
    bnb_4bit_compute_dtype=torch.bfloat16,
    # llm_int8_enable_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # torch_dtype=torch.bfloat16,
    torch_dtype="auto",
    # torch_dtype=torch.int8,
    quantization_config=bnb_config,
    # device_map="auto",
    device_map=0,
    # device_map="auto",
    # low_cpu_mem_usage=True,
    # use_cache=False
)

# model.to(device)  # Explicitly move model to CPU

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True )

# Tokens limit for model's response
MAX_NEW_TOKENS = 512

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
    # {
    #     "role": "system",
    #     "content": "You are a knowledgeable, efficient, and direct AI assistant. You provide concise answers, focusing on the key information needed. You offer suggestions tactfully when appropriate to improve outcomes. You engage in productive collaboration with the user."
    # }
    # {
    #     "role": "system",
    #     "content": "You are a knowledgeable, efficient, and direct AI assistant. You provide concise answers, focusing on the key information needed. You offer suggestions tactfully when appropriate to improve outcomes. You engage in productive collaboration with the other AI assistants."
    # }
    {
        "role": "system",
        "content": (
                    "You are a software developer AI assistent. " 
                    "Part of the code will be given, with a keyword [AUTOCOMPLETE_HERE] in the code. "
                    "You will need to analyse the entire code and understand its purpose. "
                    "Replace [AUTOCOMPLETE_HERE] with relevant code and return only the missing code. "
                    "If you think there is nothing to add, return <NOTHING_TO_ADD>. "
                    "You don't explain the code, don't summarize, you just return the missing code."
                    )
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


# load a chat history
# while True:
#     chat_history_user_file = input("Your chat history file: ").strip()
#     if chat_history_user_file:
#         if os.path.exists(chat_history_user_file):
#             with open(chat_history_user_file, "r", encoding="utf-8") as file:
#                 data = json.load(file)
#                 chat_history = data

#                 for item in data:
#                     if "time" in item:
#                         item.pop("time")

#                 chat_history = data
#                 break
#     else:
#         break


# load system role
while True:
    system_role_file = input("System role: ").strip()
    if system_role_file:
        if os.path.exists(system_role_file):
            with open(system_role_file, "r", encoding="utf-8") as file:
                chat_history[0]["content"] = file.read()

            break
    else:
        break



copied_chat_history = copy.deepcopy(chat_history)

# print(chat_history)



# Function to save the chat history to a file
def save_chat_history():
    with open(chat_history_file, "w") as file:
        json.dump(copied_chat_history, file, indent=4)

# Function to get a response
def chat_with_bot(user_input, continue_chat=True):

    global copied_chat_history

    temp_chat_history = []

    if continue_chat:
        # Append the user message to the chat history and copied chat history
        chat_history.append({"role": "user", "content": user_input})
        copied_chat_history.append({"role": "user", "content": user_input})
    else:
        temp_chat_history = copy.deepcopy(chat_history)
        temp_chat_history.append({"role": "user", "content": user_input})

    print(temp_chat_history)

    inputs = tokenizer.apply_chat_template(
      chat_history if continue_chat else temp_chat_history,
      tokenize=True,
      add_generation_prompt=True,
      return_tensors="pt",
      return_dict=True,
    ).to(model.device)

    generation_kwargs = dict(**inputs, 
        do_sample=True,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.2,
        streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
    outputs = ""

    t1 = time.time()
    thread.start()

    print("Assistant: ", end='')
    for new_text in streamer:
        outputs += new_text
        print(new_text, end='', flush=True)

    t2 = time.time()

    # bot_response = outputs[0]["generated_text"][-1]['content']
    bot_response = outputs

    # print(f"\nAssistant ({t2 - t1:.2f}s): {bot_response}")
    print(f"\nResponse in {t2 - t1}s\n")
    
    if continue_chat:
        # Add the bot's response to the chat history and copied chat history
        chat_history.append({"role": "assistant", "content": bot_response})
        copied_chat_history.append({"role": "assistant", "content": bot_response, "time": t2 - t1})
    else:
        copied_chat_history = copy.deepcopy(temp_chat_history)
        temp_chat_history.append({"role": "assistant", "content": bot_response})
        copied_chat_history.append({"role": "assistant", "content": bot_response, "time": t2 - t1})
    
    # Save the updated chat history
    save_chat_history()

    # save the last bost respone in a text file
    with open("bot_response.txt", 'w', encoding='utf-8') as file:
        file.write(f"AI assistant: {bot_response}")
    
    return bot_response

# Example usage
while True:
    user_input = input("You: ").strip()
    if user_input:
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye, friend!")
            break

        if os.path.exists(user_input):
            with open(user_input, 'r', encoding='utf-8') as file:
                user_input = file.read()


        bot_response = chat_with_bot(user_input, continue_chat=False)
