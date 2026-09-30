from encoder_model.inference import predict
from encoder_model.model import DinoClassifier
from transformers import AutoImageProcessor
import torch
from pathlib import Path
from PIL import Image
import time

model_dir = Path("models/v3")

model = DinoClassifier(n_hid=384, n_classes=10)
model.load_state_dict(torch.load(model_dir / "model.pth", map_location=torch.device("cpu")))
model.eval()

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

for i in range(1, 10):
    image_path = Path(f"data/sudoku/{i}.jpg")
    image = Image.open(image_path)
    start_time = time.time()
    print(f"Predicting number for image: {image_path}")
    print(f"Predicted number: {predict(image, model, processor)}")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print()
    

image_path = Path(f"data/eight_slavic.jpg")
image = Image.open(image_path)
print(f"Predicting number for image: {image_path}")
print(f"Predicted number and logits: {predict(image, model, processor)}")