import torch

@torch.no_grad()
def predict(image, model, processor):  
    processed_image = processor(image, return_tensors="pt")
    logits = model(processed_image)
    probabilities = torch.softmax(logits, dim=1)
    return logits.argmax(dim=1).item(), probabilities