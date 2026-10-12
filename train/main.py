import torch
import torch.nn as nn
from transformers import AutoImageProcessor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from encoder_model.model import DinoClassifier
from train.train import train

EPOCHS = 5
LR = 1e-4
BATCH_SIZE = 32

train_data_dir = Path("data/dataset/train")
val_data_dir = Path("data/dataset/val")
model_dir = Path("models/v3")
model_version = "v5"

def collate_fn(batch):
    images, labels = zip(*batch)

    return list(images), torch.tensor(labels)

def main():
    train_dataset = ImageFolder(train_data_dir, transform=None)
    val_dataset = ImageFolder(val_data_dir, transform=None)

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("Train and validation datasets must use the same class-to-index mapping")

    class_counts = torch.bincount(
        torch.tensor(train_dataset.targets),
        minlength=len(train_dataset.classes),
    )
    if torch.any(class_counts == 0):
        missing_classes = [
            train_dataset.classes[index]
            for index, count in enumerate(class_counts)
            if count == 0
        ]
        raise ValueError(f"Training dataset is missing classes: {missing_classes}")

    sample_weights = 1.0 / class_counts[train_dataset.targets].double()
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("Train dataset")
    print(len(train_dataset))
    print(train_dataset.classes)
    print(train_dataset.class_to_idx)
    print({
        train_dataset.classes[index]: count.item()
        for index, count in enumerate(class_counts)
    })

    print()
    
    print("Val Dataset")
    print(len(val_dataset))
    print(val_dataset.classes)
    print(val_dataset.class_to_idx)
    print({
        val_dataset.classes[index]: val_dataset.targets.count(index)
        for index in range(len(val_dataset.classes))
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoClassifier(n_hid=384, n_classes=len(train_dataset.classes)).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pth", map_location=device))
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.encoder.encoder.layer[-1].parameters():
        param.requires_grad = True

    for param in model.head.parameters():
        param.requires_grad = True
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.shape}")
    
    best_checkpoint = train(
        model,
        optimizer,
        processor,
        loss_fn,
        EPOCHS,
        train_loader,
        val_loader,
        device,
        train_dataset.classes,
    )

    model.load_state_dict(best_checkpoint.model_state)
    
    torch.save(model.state_dict(), f"models/{model_version}/model.pth")

if __name__ == "__main__":
    main()