import torch

@torch.no_grad()
def predict(image, model, processor):  
    processed_image = processor(image, return_tensors="pt")
    
    logits = model(processed_image)
    softmax_logits = torch.softmax(logits, dim=1)    
        
    return logits.argmax(dim=1).item(), softmax_logits