#------------------------------------------------------------
# CATEGORY METRICS
#------------------------------------------------------------
import torch


def classification_metrics(confusion, classes):
    """Explain correct and wrong predictions for each category."""
    matrix = confusion.to(torch.float64)
    correct = matrix.diag()
    support = matrix.sum(1)
    precision = correct / matrix.sum(0).clamp(min=1)
    recall = correct / support.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    # Include categories present as targets or predictions in the macro average.
    active = (support + matrix.sum(0)) > 0
    return {
        "accuracy": (correct.sum() / matrix.sum().clamp(min=1)).item(),
        "macro_f1": f1[active].mean().item() if active.any() else 0.0,
        "per_class": {name: {"precision": precision[i].item(), "recall": recall[i].item(),
                             "f1": f1[i].item(), "support": int(support[i])}
                      for i, name in enumerate(classes)},
        "confusion_matrix": confusion.tolist(),
        "class_order": classes,
        "confusion_matrix_axes": "rows = true category, columns = predicted category",
    }


@torch.inference_mode()
def evaluate_model(model, loader, device, classes):
    """Measure the classifier on images that are not used for learning."""
    model.eval()
    confusion = torch.zeros((len(classes), len(classes)), dtype=torch.long)
    total_loss, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        count += len(labels)
        indices = (labels * len(classes) + logits.argmax(1)).cpu()
        confusion += torch.bincount(indices, minlength=len(classes) ** 2).reshape_as(confusion)
    result = classification_metrics(confusion, classes)
    result["loss"] = total_loss / count
    return result
