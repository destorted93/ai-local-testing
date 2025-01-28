import time
import os
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

model_id = "Qwen/Qwen2-VL-7B-Instruct"

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype="auto", 
    device_map="auto"
)
# default processer
processor = AutoProcessor.from_pretrained(model_id)


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
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

    messages[0]["content"][1]["text"] = user_input_text

    t1 = time.time()
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=resized_image,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    t2 = time.time()

    print(f"Output ({t2 - t1}s): {output_text}")
