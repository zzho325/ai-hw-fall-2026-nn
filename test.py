import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from data import DEFAULT_DATA_ROOT, choose_device, load_images
from models import MODELS


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved models on the MNIST test set.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    device = choose_device(args.device)
    torch.set_num_threads(4)
    images, labels = load_images(args.data_root, train=False)
    results = args.output / "results"
    results.mkdir(parents=True, exist_ok=True)
    training = json.loads((results / "training_summary.json").read_text())
    selected_name = max(training["runs"], key=lambda run: run["best_validation_accuracy"])["model"]
    report = {"test_examples": len(labels), "runs": []}

    for run in training["runs"]:
        name = run["model"]
        checkpoint = torch.load(
            args.output / "checkpoints" / f"{name}.pt",
            map_location="cpu",
            weights_only=True,
        )
        model = MODELS[name]().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        normalization = checkpoint["normalization"]
        predictions = []
        loss_sum = torch.zeros((), device=device)
        with torch.inference_mode():
            for batch, targets in zip(
                images.split(args.batch_size),
                labels.split(args.batch_size),
            ):
                batch = (batch.to(device) - normalization["mean"]) / normalization["std"]
                logits = model(batch)
                loss_sum += F.cross_entropy(logits, targets.to(device), reduction="sum")
                predictions.append(logits.argmax(1).cpu())
        predictions = torch.cat(predictions)
        correct = int((predictions == labels).sum())
        report["runs"].append({
            "model": name,
            "selected_epoch": checkpoint["epoch"],
            "validation_accuracy": checkpoint["validation_accuracy"],
            "correct": correct,
            "accuracy": correct / len(labels),
            "cross_entropy_loss": loss_sum.item() / len(labels),
        })
        print(f"{name}: {correct}/{len(labels)} correct ({100 * correct / len(labels):.2f}%)")

        if name == selected_name:
            matrix = torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
            fig, ax = plt.subplots(figsize=(7, 6))
            image = ax.imshow(matrix.numpy(), cmap="Blues")
            for row in range(10):
                for col in range(10):
                    ax.text(
                        col,
                        row,
                        str(matrix[row, col].item()),
                        ha="center",
                        va="center",
                        color="white" if matrix[row, col] > matrix.max() / 2 else "black",
                        fontsize=8,
                    )
            ax.set(
                title=f"{name}: {100 * correct / len(labels):.2f}% test accuracy",
                xlabel="Predicted digit",
                ylabel="True digit",
                xticks=range(10),
                yticks=range(10),
            )
            fig.colorbar(image, ax=ax, label="Test examples")
            fig.tight_layout()
            fig.savefig(results / "confusion_matrix.png", dpi=150)
            plt.close(fig)

    (results / "test_results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
