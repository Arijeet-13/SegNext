"""
Plotting utilities for semantic segmentation training with HF Trainer.

Consumes `trainer.state.log_history` which is a list of dicts with keys like:
  - Training:  loss, learning_rate, epoch, step
  - Eval:      eval_loss, eval_mIoU, eval_Safe_mIoU, eval_mIoU_Main_Classes,
               eval_Safe_mIoU_Main_Classes, epoch, step
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _setup_style():
    """Consistent publication-quality style."""
    sns.set_style("white")
    sns.set_context("paper")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })


def extract_metrics(log_history):
    """
    Parse trainer.state.log_history into separate lists.

    Returns a dict with:
        train_loss, train_epoch        – per logging step
        eval_loss, eval_epoch          – per eval epoch
        eval_mIoU, eval_Safe_mIoU      – per eval epoch
        eval_mIoU_Main, eval_Safe_mIoU_Main
        learning_rate
    """
    metrics = {
        "train_loss": [], "train_epoch": [],
        "learning_rate": [], "lr_step": [],
        "eval_loss": [], "eval_epoch": [],
        "eval_mIoU": [], "eval_Safe_mIoU": [],
        "eval_mIoU_Main": [], "eval_Safe_mIoU_Main": [],
    }

    for entry in log_history:
        # Training log entries
        if "loss" in entry and "eval_loss" not in entry:
            metrics["train_loss"].append(entry["loss"])
            metrics["train_epoch"].append(entry.get("epoch", 0))
        if "learning_rate" in entry:
            metrics["learning_rate"].append(entry["learning_rate"])
            metrics["lr_step"].append(entry.get("step", 0))

        # Eval log entries
        if "eval_loss" in entry:
            metrics["eval_loss"].append(entry["eval_loss"])
            metrics["eval_epoch"].append(entry.get("epoch", 0))
            metrics["eval_mIoU"].append(entry.get("eval_mIoU", float("nan")))
            metrics["eval_Safe_mIoU"].append(entry.get("eval_Safe_mIoU", float("nan")))
            metrics["eval_mIoU_Main"].append(entry.get("eval_mIoU_Main_Classes", float("nan")))
            metrics["eval_Safe_mIoU_Main"].append(entry.get("eval_Safe_mIoU_Main_Classes", float("nan")))

    return metrics


def plot_loss(metrics, plot_dir):
    """Training loss vs. validation loss over epochs."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.plot(metrics["train_epoch"], metrics["train_loss"],
            label="Train Loss", color="royalblue", linewidth=1.5, alpha=0.7)
    ax.plot(metrics["eval_epoch"], metrics["eval_loss"],
            label="Val Loss", color="darkorange", linewidth=2, marker="o", markersize=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(False)
    fig.savefig(os.path.join(plot_dir, "loss.png"), bbox_inches="tight")
    plt.close(fig)


def plot_miou(metrics, plot_dir):
    """mIoU and Safe mIoU over epochs."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    epochs = metrics["eval_epoch"]
    ax.plot(epochs, metrics["eval_mIoU"],
            label="mIoU", color="seagreen", linewidth=2, marker="o", markersize=5)
    ax.plot(epochs, metrics["eval_Safe_mIoU"],
            label="Safe mIoU", color="crimson", linewidth=2, marker="s", markersize=5)
    ax.plot(epochs, metrics["eval_mIoU_Main"],
            label="mIoU (Main Classes)", color="seagreen", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.plot(epochs, metrics["eval_Safe_mIoU_Main"],
            label="Safe mIoU (Main Classes)", color="crimson", linewidth=1.5, linestyle="--", alpha=0.6)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("Validation mIoU & Safe mIoU")
    ax.legend(fontsize=10)
    ax.grid(False)
    fig.savefig(os.path.join(plot_dir, "miou.png"), bbox_inches="tight")
    plt.close(fig)


def plot_learning_rate(metrics, plot_dir):
    """Learning rate schedule over training steps."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

    ax.plot(metrics["lr_step"], metrics["learning_rate"],
            color="mediumpurple", linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(False)
    fig.savefig(os.path.join(plot_dir, "learning_rate.png"), bbox_inches="tight")
    plt.close(fig)


def plot_per_class_iou(ious, safe_ious, class_names, plot_dir):
    """
    Horizontal bar chart of per-class IoU and Safe IoU.

    Args:
        ious: list of per-class IoU values (length = num_classes).
        safe_ious: list of per-class Safe IoU values.
        class_names: list of class name strings.
        plot_dir: directory to save the plot.
    """
    _setup_style()
    n = len(class_names)
    y = np.arange(n)
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.4)), dpi=300)
    ax.barh(y - bar_h / 2, [v * 100 for v in ious], bar_h,
            label="IoU", color="steelblue")
    ax.barh(y + bar_h / 2, [v * 100 for v in safe_ious], bar_h,
            label="Safe IoU", color="indianred")

    ax.set_yticks(y)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("IoU (%)")
    ax.set_title("Per-Class IoU & Safe IoU")
    ax.legend(loc="lower right")
    ax.grid(False)
    ax.invert_yaxis()
    fig.savefig(os.path.join(plot_dir, "per_class_iou.png"), bbox_inches="tight")
    plt.close(fig)


def plot_all(log_history, plot_dir, per_class_ious=None, per_class_safe_ious=None, class_names=None):
    """
    Generate all plots from Trainer log history.

    Args:
        log_history: trainer.state.log_history (list of dicts).
        plot_dir: directory to save plots.
        per_class_ious: optional list of per-class IoU from final eval.
        per_class_safe_ious: optional list of per-class Safe IoU.
        class_names: optional list of class name strings.
    """
    os.makedirs(plot_dir, exist_ok=True)
    metrics = extract_metrics(log_history)

    if metrics["train_loss"]:
        plot_loss(metrics, plot_dir)
        print(f"  Saved loss.png")

    if metrics["eval_mIoU"]:
        plot_miou(metrics, plot_dir)
        print(f"  Saved miou.png")

    if metrics["learning_rate"]:
        plot_learning_rate(metrics, plot_dir)
        print(f"  Saved learning_rate.png")

    if per_class_ious is not None and class_names is not None:
        safe = per_class_safe_ious if per_class_safe_ious is not None else [0.0] * len(per_class_ious)
        plot_per_class_iou(per_class_ious, safe, class_names, plot_dir)
        print(f"  Saved per_class_iou.png")

    print(f"All plots saved to {plot_dir}")