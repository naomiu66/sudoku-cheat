import torch
from train.best_checkpoint import BestCheckpoint


@torch.no_grad()
def classification_report(model, processor, data_loader, device, class_names):
    model.eval()
    class_count = len(class_names)
    confusion = torch.zeros((class_count, class_count), dtype=torch.int64)

    for images, labels in data_loader:
        processed_input = processor(images, return_tensors="pt")
        processed_input = {
            key: value.to(device)
            for key, value in processed_input.items()
        }
        predictions = model(processed_input).argmax(dim=1).cpu()
        labels = labels.cpu()
        confusion += torch.bincount(
            labels * class_count + predictions,
            minlength=class_count * class_count,
        ).reshape(class_count, class_count)

    support = confusion.sum(dim=1).to(torch.float64)
    predicted_count = confusion.sum(dim=0).to(torch.float64)
    true_positive = confusion.diag().to(torch.float64)
    precision = torch.where(predicted_count > 0, true_positive / predicted_count.clamp_min(1), 0)
    recall = torch.where(support > 0, true_positive / support.clamp_min(1), 0)
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall).clamp_min(1e-12),
        0,
    )
    total = support.sum()
    accuracy = true_positive.sum() / total.clamp_min(1)
    macro = torch.stack((precision.mean(), recall.mean(), f1.mean()))
    weighted = torch.stack((
        (precision * support).sum() / total.clamp_min(1),
        (recall * support).sum() / total.clamp_min(1),
        (f1 * support).sum() / total.clamp_min(1),
    ))

    print("\nClassification report (validation):")
    print(f"{'class':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    for index, name in enumerate(class_names):
        print(
            f"{name:>12} {precision[index].item():>10.2f} "
            f"{recall[index].item():>10.2f} {f1[index].item():>10.2f} "
            f"{int(support[index].item()):>10}"
        )
    print(
        f"{'accuracy':>12} {'':>10} {'':>10} {accuracy.item():>10.2f} "
        f"{int(total.item()):>10}"
    )
    print(
        f"{'macro avg':>12} {macro[0].item():>10.2f} {macro[1].item():>10.2f} "
        f"{macro[2].item():>10.2f} {int(total.item()):>10}"
    )
    print(
        f"{'weighted avg':>12} {weighted[0].item():>10.2f} "
        f"{weighted[1].item():>10.2f} {weighted[2].item():>10.2f} "
        f"{int(total.item()):>10}"
    )


@torch.no_grad()
def val(model, processor, val_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in val_loader:
        labels = labels.to(device)

        processed_input = processor(images, return_tensors="pt")
        processed_input = {
            k: v.to(device)
            for k, v in processed_input.items()
        }
        
        logits = model(processed_input)
        preds = logits.argmax(dim=1)
        loss = loss_fn(logits, labels)
        
        batch_size = labels.size(0)

        correct += (preds == labels).sum().item()        
        total_loss += loss.item() * batch_size
        total += batch_size
    return total_loss / total, correct / total
    

def train(model, optimizer, processor, loss_fn, epochs, train_loader, val_loader, device, class_names):
    best_checkpoint = BestCheckpoint(model)
    for epoch in range(epochs):

        total_loss = 0.0
        correct = 0
        total = 0

        model.train()
        
        for images, labels in train_loader:
            labels = labels.to(device)
            processed_input = processor(images, return_tensors="pt")
            processed_input = {
                k: v.to(device)
                for k, v in processed_input.items()
            }

            logits = model(processed_input)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            correct += (preds == labels).sum().item()
            total += batch_size
        
        
        train_loss = total_loss / total
        train_accuracy = correct / total
        test_loss, test_acc = val(model, processor, val_loader, loss_fn, device)
        best_checkpoint.update(model, test_acc)

        print(
            f"[EPOCH] {epoch + 1}/{epochs}"
            f" | test_acc: {test_acc:.4f}" 
            f" | test_loss: {test_loss:.4f}" 
            f" | train_acc: {train_accuracy:.4f}"
            f" | train_loss: {train_loss:.4f}"
        )
    model.load_state_dict(best_checkpoint.model_state)
    classification_report(model, processor, val_loader, device, class_names)
    return best_checkpoint

