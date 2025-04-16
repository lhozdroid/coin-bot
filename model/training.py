import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, StepLR


def get_scheduler(optimizer, config):
    """
    Configures the learning rate scheduler

    Args:
        optimizer:
        config:

    Returns:
        Learning rate scheduler
    """
    sched_cfg = config.get("scheduler", {})
    sched_type = sched_cfg.get("type", "cosine")

    if sched_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=sched_cfg.get("t_max", 10), eta_min=float(sched_cfg.get("eta_min", 1e-5)))
    elif sched_type == "cosine_warm_restart":
        return CosineAnnealingWarmRestarts(optimizer, T_0=sched_cfg.get("t_0", 10), T_mult=sched_cfg.get("t_mult", 2), eta_min=float(sched_cfg.get("eta_min", 1e-5)))
    elif sched_type == "plateau":
        return ReduceLROnPlateau(optimizer, patience=5)
    elif sched_type == "step":
        return StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        return None


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for a single epoch

    Args:
        model:
        dataloader:
        optimizer:
        criterion:
        device:

    Returns:
        Tuple containing average loss and accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)  # [batch_size, num_classes]
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on a validation or test set

    Args:
        model:
        dataloader:
        criterion:
        device:

    Returns:
        Tuple with average loss, accuracy, predictions, and labels
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            total_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels


def train_model(model, train_loader, val_loader, test_loader, config, on_epoch_end=None):
    """
    Training loop for the full training process
    Args:
        model:
        train_loader:
        val_loader:
        test_loader:
        config:
        on_epoch_end:

    Returns:

    """
    device = torch.device(config.get("device", "cuda"))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config.get("learning_rate", 1e-3))
    scheduler = get_scheduler(optimizer, config)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = config.get("epochs", 50)

    best_val_loss = float("inf")
    early_stop_counter = 0
    patience = config.get("early_stopping", {}).get("patience", 5)
    delta = config.get("early_stopping", {}).get("delta", 1e-4)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler:
            scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if on_epoch_end:
            on_epoch_end(epoch, val_loss, val_acc)

        if val_loss < best_val_loss - delta:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    print("\nEvaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
