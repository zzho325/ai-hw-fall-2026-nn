import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data import DEFAULT_DATA_ROOT, choose_device, load_images, training_split
from models import MODELS


def main():
    parser = argparse.ArgumentParser(description="Train MNIST digit classifiers.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")

    device = choose_device(args.device)
    torch.set_num_threads(4)
    images, labels = load_images(args.data_root, train=True)
    train_indices, val_indices, split = training_split(args.seed, len(images))
    train_images = images[train_indices].to(device)
    train_labels = labels[train_indices].to(device)
    val_images = images[val_indices].to(device)
    val_labels = labels[val_indices].to(device)
    mean = train_images.mean().item()
    std = train_images.std().item()
    val_images = (val_images - mean) / std
    (args.output / "checkpoints").mkdir(parents=True, exist_ok=True)
    results = args.output / "results"
    results.mkdir(parents=True, exist_ok=True)
    summary = {"split": split, "runs": []}

    for name in args.models:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        model = MODELS[name]().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        best_correct = -1
        best_loss = float("inf")
        history = []
        print(f"Training {name} on {device}", flush=True)

        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(train_images), generator=generator)
            train_loss = torch.zeros((), device=device)
            train_correct = torch.zeros((), device=device, dtype=torch.long)
            for indices in order.split(args.batch_size):
                indices = indices.to(device)
                batch = train_images[indices]
                targets = train_labels[indices]
                if name == "cnn_aug":
                    count = len(batch)
                    angle = (torch.rand(count, device=device) * 20 - 10) * torch.pi / 180
                    translations = torch.rand(count, 2, device=device) * 0.32 - 0.16
                    theta = torch.zeros(count, 2, 3, device=device)
                    theta[:, 0, 0] = angle.cos()
                    theta[:, 0, 1] = -angle.sin()
                    theta[:, 1, 0] = angle.sin()
                    theta[:, 1, 1] = angle.cos()
                    theta[:, :, 2] = translations
                    grid = F.affine_grid(theta, batch.size(), align_corners=False)
                    batch = F.grid_sample(
                        batch,
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    )
                batch = (batch - mean) / std
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.detach() * len(targets)
                train_correct += (logits.argmax(1) == targets).sum()

            model.eval()
            val_loss = torch.zeros((), device=device)
            val_correct = torch.zeros((), device=device, dtype=torch.long)
            with torch.inference_mode():
                for batch, targets in zip(
                    val_images.split(args.batch_size),
                    val_labels.split(args.batch_size),
                ):
                    logits = model(batch)
                    val_loss += F.cross_entropy(logits, targets, reduction="sum")
                    val_correct += (logits.argmax(1) == targets).sum()
            correct = val_correct.item()
            average_val_loss = val_loss.item() / len(val_images)
            row = {
                "epoch": epoch,
                "train_loss": train_loss.item() / len(train_images),
                "train_accuracy": train_correct.item() / len(train_images),
                "validation_loss": average_val_loss,
                "validation_accuracy": correct / len(val_images),
            }
            history.append(row)
            print(f"{name} epoch {epoch}: {json.dumps(row)}", flush=True)
            if correct > best_correct or (
                correct == best_correct and average_val_loss < best_loss
            ):
                best_correct, best_loss = correct, average_val_loss
                checkpoint = {
                    "model_name": name,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "validation_accuracy": correct / len(val_images),
                    "validation_loss": average_val_loss,
                    "normalization": {"mean": mean, "std": std},
                    "split": split,
                    "hyperparameters": {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "optimizer": "Adam",
                        "augmentation": name == "cnn_aug",
                    },
                }
                torch.save(checkpoint, args.output / "checkpoints" / f"{name}.pt")

        with (results / f"history_{name}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=history[0])
            writer.writeheader()
            writer.writerows(history)
        summary["runs"].append({
            "model": name,
            "best_epoch": checkpoint["epoch"],
            "best_validation_accuracy": checkpoint["validation_accuracy"],
            "hyperparameters": checkpoint["hyperparameters"],
        })
        (results / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Training complete. Run test.py for held-out test evaluation.", flush=True)


if __name__ == "__main__":
    main()
