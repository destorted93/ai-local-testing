import time
import os
from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # "min_pixels": 100,
                # "max_pixels": 1000
                },
            {
                "type": "text", 
                "text": "Describe this image."
            },
        ],
    }
]

while True:
    user_input_image = input("Image location: ")
    if user_input_image == "exit":
        break
    if user_input_image.strip() == '':
        continue
    if not os.path.exists(user_input_image):
        print(f"{user_input_image} does not exist!")
        continue

    user_input_text = input("Text: ")
    if user_input_text == "exit":
        break
    if user_input_text.strip() == '':
        continue



    # Image
    url = user_input_image
    image = Image.open(url)

    IMAGE_MAX_SIZE = 1000

    # Get original dimensions
    original_width, original_height = image.size

    # Check dimensions and resize accordingly
    if original_width <= IMAGE_MAX_SIZE and original_height <= IMAGE_MAX_SIZE:
        # Both dimensions are within the limit, no resizing needed
        resized_image = image
    elif original_width > IMAGE_MAX_SIZE or original_height > IMAGE_MAX_SIZE:
        # Resize based on the largest dimension
        if original_width > original_height:
            # Width is the largest dimension
            new_width = IMAGE_MAX_SIZE
            new_height = int((new_width / original_width) * original_height)
        else:
            # Height is the largest dimension
            new_height = 1000
            new_width = int((new_height / original_height) * original_width)
        
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    resized_image.save("resized_image.jpg")


    # update the conversation
    conversation[0]["content"][1]["text"] = user_input_text

    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

    inputs = processor(
        text=[text_prompt], images=[resized_image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")


    t1 = time.time()
    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    t2 = time.time()

    print(f"Output ({t2 - t1}s): {output_text}")
