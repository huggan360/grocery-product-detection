#------------------------------------------------------------
# TRAIN THE VISION TRANSFORMER ON PRODUCT CROPS
#------------------------------------------------------------
import argparse
import warnings

import torch
from torch import nn

from production.vit import ARCHITECTURE, IMAGE_SIZE, head_parameters, initialize_classifier
from utils.common import choose_device, load_config, new_run_folder, seed_everything, write_json
from training.data import ProductDataset, make_loader
from training.metrics import evaluate_model


def train_classifier(config):
    """Fine-tune the ViT and save the model that works best on validation images."""
    settings = config["classifier"]
    if settings["epochs"] < 1 or settings["patience"] < 1:
        raise ValueError("Epochs and patience must be positive.")
    seed_everything(config["seed"])
    device = choose_device(config["device"])
    from pathlib import Path
    data = Path(settings["data"])
    train_data = ProductDataset(data / "train", training=True)
    val_data = ProductDataset(data / "val", classes=train_data.classes)
    missing = set(range(len(train_data.classes))) - {label for _, label in val_data.samples}
    if missing:
        warnings.warn("Validation has no examples for: " + ", ".join(train_data.classes[i] for i in sorted(missing))
                      + ". Metrics cannot measure recall for these categories.", stacklevel=2)
    train_loader = make_loader(train_data, settings["batch_size"], settings["workers"],
                               training=True, balanced=settings["balanced_sampling"],
                               seed=config["seed"])
    val_loader = make_loader(val_data, settings["batch_size"], settings["workers"])
    output = new_run_folder(settings["output"])
    write_json(output / "config.json", config)
    write_json(output / "classes.json", train_data.classes)
    model = initialize_classifier(train_data.classes, settings.get("initialize_from")).to(device)

    # The old body learns slowly. The new category layer learns faster.
    head = head_parameters(model)
    backbone = [p for p in model.parameters() if all(p is not h for h in head)]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": settings["learning_rate"]},
        {"params": head, "lr": settings["head_learning_rate"]},
    ], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])
    use_amp = settings["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_function = nn.CrossEntropyLoss()
    best_score, stale_epochs, history = -1.0, 0, []

    #------------------------------------------------------------
    # LEARN FROM TRAINING IMAGES, THEN CHECK VALIDATION IMAGES
    #------------------------------------------------------------
    for epoch in range(settings["epochs"]):
        frozen = epoch < settings["freeze_backbone_epochs"]
        for parameter in backbone:
            parameter.requires_grad_(not frozen)
        model.train()
        if frozen:
            # Keep dropout/normalization of the frozen body in inference mode.
            for name, module in model.named_children():
                if name != "head":
                    module.eval()
        loss_sum, count = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_function(model(images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
        scheduler.step()
        metrics = evaluate_model(model, val_loader, device, train_data.classes)
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / count, "validation": metrics})
        write_json(output / "history.json", history)
        checkpoint = {
            "format_version": 2, "architecture": ARCHITECTURE, "image_size": IMAGE_SIZE,
            "model": model.state_dict(), "classes": train_data.classes,
            "epoch": epoch + 1, "validation": metrics, "config": config,
        }
        torch.save(checkpoint, output / "last.pt")
        if metrics["macro_f1"] > best_score:
            best_score, stale_epochs = metrics["macro_f1"], 0
            torch.save(checkpoint, output / "best.pt")
        else:
            stale_epochs += 1
        print(f"Epoch {epoch + 1}: train loss={loss_sum / count:.4f}, "
              f"val accuracy={metrics['accuracy']:.3f}, val macro F1={metrics['macro_f1']:.3f}")
        if stale_epochs >= settings["patience"]:
            print("Stopping because validation macro F1 stopped improving.")
            break
    return output / "best.pt"


#------------------------------------------------------------
# RUN THIS FILE ON ITS OWN
#------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the pretrained product ViT.")
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()
    print(train_classifier(load_config(args.config)))
