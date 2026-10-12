"""Name the held object with GLM-4.1V; the name becomes the SAM3 text prompt."""
import gc
import re

import torch
from PIL import Image
from transformers import AutoProcessor, Glm4vForConditionalGeneration


def main(file_path, text_prompt):
    MODEL_PATH = "THUDM/GLM-4.1V-9B-Thinking"
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # transformers>=5 cannot fetch local paths given as image URLs; pass a PIL image.
    image = Image.open(file_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": f"{text_prompt}"
                }
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=8192, do_sample=False)
    output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    del model, inputs, generated_ids
    gc.collect()
    torch.cuda.empty_cache()
    match = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", output_text)
    return match.group(1).strip() if match else None
