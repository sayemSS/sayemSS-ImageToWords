from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import time


def generate_image_description(image_path):
    try:
        # 📷 Open the image
        image = Image.open(image_path).convert("RGB")

        # 📦 Load the model and processor
        print("Loading model and processor...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("Model and processor loaded.")

        # 🚀 Send the model to CPU or GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # 🧑‍💻 Preprocess the image
        inputs = processor(images=image, return_tensors="pt").to(device)
        print(f"Inputs: {inputs}")  # Check inputs

        # 📜 Generate caption
        time.sleep(5)  # 5 seconds delay to allow model to load
        outputs = model.generate(**inputs, num_beams=4, min_length=10)
        print(f"Outputs: {outputs}")  # Check outputs

        # Decode and print output
        description = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"\n✅ Generated Caption: {description}\n")

        return description

    except Exception as e:
        print(f"Error: {e}")


# 🎯 Path for the image (in the same folder as the script)
image_path = "Untitled.jpeg"  # Only the image name
description = generate_image_description(image_path)
