import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from datalib.idd_dataset import IDDDataset
from models import load_segformer_b3, load_segnext_large
from utils.plot import plot_all
from utils.safe_iou import L3_CLASSES, SemanticHierarchy, fast_hist, per_class_iu

# --------------------------------------------------
# Safe IoU hierarchy setup
# --------------------------------------------------
hierarchy = SemanticHierarchy()
hierarchy.initialize_hierarchy()
hierarchy.generate_N3_classes(L3_CLASSES)

IMP_CLASSES = [
    "person", "rider", "motorcycle", "bicycle",
    "autorickshaw", "car", "truck", "bus"
]

ind_of_imp_classes = hierarchy.get_important_classes(
    IMP_CLASSES, L3_CLASSES
)
class_mapping = hierarchy.class_to_id_mapping(L3_CLASSES)

# --------------------------------------------------
# Optional loss functions you might want to use later.
# They are kept here as commented templates so the current
# training behavior remains exactly the same.
#
# """ COMMENTED OUT FOCAL LOSS """
# # FOCAL LOSS:
# def focal_loss(
#     logits,
#     targets,
#     alpha: float = 0.25,
#     gamma: float = 2.0,
#     ignore_index: int = 255,
# ):
#     """
#     Multi-class focal loss on [N, C, H, W] logits and [N, H, W] targets.
#     Compatible with ignore_index (e.g. 255 for IDD-AW).
#     """
#     ce = F.cross_entropy(
#         logits,
#         targets,
#         ignore_index=ignore_index,
#         reduction="none",
#     )  # [N, H, W]
# 
#     valid_mask = (targets != ignore_index).float()  # [N, H, W]
# 
#     # p_t = exp(-CE)
#     p_t = torch.exp(-ce)
# 
#     # Focal weighting
#     focal_weight = alpha * (1.0 - p_t) ** gamma
# 
#     loss = focal_weight * ce
#     loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-10)
#     return loss
#
# SOFT DICE LOSS:
# def soft_dice_loss(
#     logits,
#     targets,
#     num_classes: int,
#     ignore_index: int = 255,
#     smooth: float = 1.0,
# ):
#     """
#     Multi-class soft Dice loss.
#     logits: [N, C, H, W], targets: [N, H, W] with class indices.
#     """
#     n, c, h, w = logits.shape
#     probs = torch.softmax(logits, dim=1)  # [N, C, H, W]
#
#     # Mask ignore_index
#     mask = (targets != ignore_index).unsqueeze(1).float()  # [N, 1, H, W]
#     probs = probs * mask
#
#     # One-hot targets (ignore_index positions are all zeros)
#     t = torch.zeros_like(probs)
#     valid_targets = targets.clone()
#     valid_targets[targets == ignore_index] = 0
#     t.scatter_(1, valid_targets.unsqueeze(1), 1.0)
#     t = t * mask
#
#     # Flatten
#     probs_flat = probs.view(n, c, -1)
#     t_flat = t.view(n, c, -1)
#
#     intersection = (probs_flat * t_flat).sum(-1)  # [N, C]
#     union = probs_flat.sum(-1) + t_flat.sum(-1)   # [N, C]
#
#     dice = (2.0 * intersection + smooth) / (union + smooth)
#     dice_loss = 1.0 - dice.mean()
#     return dice_loss
#
# FOCAL + DICE LOSS:
# def focal_dice_loss(
#     logits,
#     targets,
#     num_classes: int,
#     alpha: float = 0.25,
#     gamma: float = 2.0,
#     ignore_index: int = 255,
#     dice_weight: float = 1.0,
#     focal_weight: float = 1.0,
# ):
#     """
#     Combined focal + Dice loss.
#     Tune dice_weight and focal_weight to balance them.
#     """
#     fl = focal_loss(
#         logits,
#         targets,
#         alpha=alpha,
#         gamma=gamma,
#         ignore_index=ignore_index,
#     )
#     dl = soft_dice_loss(
#         logits,
#         targets,
#         num_classes=num_classes,
#         ignore_index=ignore_index,
#     )
#     return focal_weight * fl + dice_weight * dl

# --------------------------------------------------
# Custom Trainer (poly scheduler + cross entropy loss)
# --------------------------------------------------
class CustomTrainer(Trainer):

    def create_optimizer_and_scheduler(self, num_training_steps):
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "norm.weight"]
        head_keywords = ["decode_head", "classifier", "segmentation_head", "head"]

        base_lr = self.args.learning_rate
        weight_decay = self.args.weight_decay

        param_groups = [
            {"params": [], "weight_decay": weight_decay, "lr": base_lr},
            {"params": [], "weight_decay": 0.0, "lr": base_lr},
            {"params": [], "weight_decay": weight_decay, "lr": base_lr * 10.0},
            {"params": [], "weight_decay": 0.0, "lr": base_lr * 10.0},
        ]

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            is_head = any(k in name for k in head_keywords)
            has_no_decay = any(nd in name for nd in no_decay)

            if is_head and has_no_decay:
                group_idx = 3
            elif is_head:
                group_idx = 2
            elif has_no_decay:
                group_idx = 1
            else:
                group_idx = 0

            param_groups[group_idx]["params"].append(param)

        param_groups = [g for g in param_groups if g["params"]]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )

        warmup_ratio = getattr(self.args, "warmup_ratio", 0.0)
        warmup_steps = int(warmup_ratio * num_training_steps)
        power = 0.9

        def lr_lambda(current_step: int):
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            if current_step >= num_training_steps:
                return 0.0

            if num_training_steps == warmup_steps:
                return 0.0

            progress = float(current_step - warmup_steps) / float(
                max(1, num_training_steps - warmup_steps)
            )
            progress = min(max(progress, 0.0), 1.0)
            return max((1.0 - progress) ** power, 1e-3)

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Loss options:
          A) Standard cross-entropy (default, used now)
          B) Focal loss             (uncomment to use)
          C) Focal + Dice loss      (uncomment to use)
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # """ ALL OTHER LOSSES COMMENTED OUT TO KEEP ONLY CROSS ENTROPY """
        loss = F.cross_entropy(logits, labels, ignore_index=255)

        # # --- CE + Focal Combined ---
        # ce_loss = F.cross_entropy(logits, labels, ignore_index=255)
        # 
        # fl_loss = focal_loss(
        #     logits,
        #     labels,
        #     alpha=0.25,
        #     gamma=2.0,
        #     ignore_index=255,
        # )
        # 
        # # Balanced combination (tune weights if needed)
        # loss = 0.5 * ce_loss + 0.5 * fl_loss

        # --- Option B: focal loss (enable by commenting out Option A) ---
        # loss = focal_loss(
        #     logits,
        #     labels,
        #     alpha=0.25,
        #     gamma=2.0,
        #     ignore_index=255,
        # )

        # --- Option C: focal + Dice loss (enable by commenting out Option A) ---
        # loss = focal_dice_loss(
        #     logits,
        #     labels,
        #     num_classes=len(L3_CLASSES),  # 26 for IDD-AW
        #     alpha=0.25,
        #     gamma=2.0,
        #     ignore_index=255,
        #     dice_weight=1.0,
        #     focal_weight=1.0,
        # )

        return (loss, outputs) if return_outputs else loss

# --------------------------------------------------
# Save best model in Hugging Face format (separate folder)
# --------------------------------------------------
class SaveBestHFCallback(TrainerCallback):

    def __init__(self, save_dir, processor):
        self.best_miou = -1.0
        self.save_dir = save_dir
        self.processor = processor
        os.makedirs(save_dir, exist_ok=True)
        
        # Read previous best score directly from HuggingFace's latest checkpoint state
        try:
            import json
            import glob
            output_dir = os.path.dirname(save_dir)
            checkpoints = [d for d in glob.glob(os.path.join(output_dir, "checkpoint-*")) if os.path.isdir(d)]
            if checkpoints:
                # Get the checkpoint with the highest step number
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                state_file = os.path.join(latest_checkpoint, "trainer_state.json")
                
                if os.path.exists(state_file):
                    with open(state_file, "r") as f:
                        state_data = json.load(f)
                        if state_data.get("best_metric") is not None:
                            self.best_miou = float(state_data["best_metric"])
                            
                            print(f"\n✅ Resumed previous best metrics from {os.path.basename(latest_checkpoint)}:")
                            print(f"   - mIoU: {self.best_miou:.4f}\n")
        except Exception as e:
            print(f"Note: Starting fresh best score tracking ({e})")
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        miou = metrics.get("eval_mIoU")
        if miou is None:
            return

        if miou > self.best_miou:
            self.best_miou = miou
            model = kwargs["model"]
            model.save_pretrained(self.save_dir)
            self.processor.save_pretrained(self.save_dir)
            print(f"\n🔥 New best mIoU: {miou:.4f} — HF model saved to {self.save_dir}\n")

# --------------------------------------------------
# Preprocess logits for metrics
# --------------------------------------------------
def preprocess_logits_for_metrics(logits, labels):

    if isinstance(logits, tuple):
        logits = logits[0]

    upsampled = F.interpolate(
        logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    return upsampled.argmax(dim=1).to(torch.uint8)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = predictions.astype(np.int64)

    num_classes_total = len(L3_CLASSES)

    hist = np.zeros((num_classes_total, num_classes_total), dtype=np.float32)
    for p, l in zip(predictions, labels):
        hist += fast_hist(
            l.flatten(),
            p.flatten(),
            num_classes_total,
        )

    ious, safe_ious = per_class_iu(
        hist,
        class_mapping,
        IMP_CLASSES,
        ind_of_imp_classes,
        num_classes=num_classes_total,
    )

    pixel_accuracy = np.diag(hist).sum() / (hist.sum() + 1e-10)
    class_acc = np.diag(hist) / (hist.sum(axis=1) + 1e-10)
    mean_class_acc = np.nanmean(class_acc)

    return {
        "mIoU": np.nanmean(ious),
        "Safe_mIoU": np.nanmean(safe_ious),
        "Pixel_Accuracy": pixel_accuracy,
        "Mean_Class_Accuracy": mean_class_acc,
        "mIoU_Main_Classes": np.nanmean(
            [ious[i] for i, c in enumerate(L3_CLASSES) if c in IMP_CLASSES]
        ),
        "Safe_mIoU_Main_Classes": np.nanmean(
            [safe_ious[i] for i, c in enumerate(L3_CLASSES) if c in IMP_CLASSES]
        ),
    }

# --------------------------------------------------
# Main training pipeline
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/segnext_large.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg["model"].get("name", "segformer-b3") == "segformer-b3":
        model, processor = load_segformer_b3(
            num_classes=cfg["model"]["num_classes"],
            checkpoint=cfg["model"].get("pretrained_checkpoint"),
            image_size=cfg["model"]["image_size"],
        )
    else:
        model, processor = load_segnext_large(
            num_classes=cfg["model"]["num_classes"],
            image_size=cfg["model"]["image_size"],
            checkpoint=cfg["model"].get("pretrained_checkpoint"),
        )

    processor.do_reduce_labels = cfg["dataset"]["reduce_labels"]

    # aug_cfg = cfg.get("augmentation", {
    #     "resize_size": [1024, 768],
    #     "crop_size": [512, 512],
    #     "cat_max_ratio": 0.75,
    #     "flip_prob": 0.5,
    # })
    # train_augmentation = get_train_augmentation(
    #     resize_size=tuple(aug_cfg["resize_size"]),
    #     crop_size=tuple(aug_cfg["crop_size"]),
    #     cat_max_ratio=aug_cfg.get("cat_max_ratio", 0.75),
    #     flip_prob=aug_cfg.get("flip_prob", 0.5),
    #     ignore_index=cfg["dataset"]["ignore_index"],
    # )

    train_dataset = IDDDataset(
        img_root=cfg["paths"]["image_root"],
        label_root=cfg["paths"]["label_root"],
        split_file=cfg["paths"]["train_split"],
        processor=processor,
        # augmentation=train_augmentation,
    )
    
    # # For validation, we DO NOT want to apply random crops or photometric distortions.
    # # The SegformerImageProcessor will natively handle resizing to the eval image_size.
    # val_augmentation = None

    val_dataset = IDDDataset(
        img_root=cfg["paths"]["image_root"],
        label_root=cfg["paths"]["label_root"],
        split_file=cfg["paths"]["val_split"],
        processor=processor,
        # augmentation=val_augmentation,
    )

    training_args = TrainingArguments(
        output_dir=cfg["paths"]["output_dir"],
        learning_rate=cfg["training"]["learning_rate"],
        per_device_train_batch_size=cfg["training"]["train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["val_batch_size"],
        num_train_epochs=cfg["training"]["epochs"],
        weight_decay=cfg["training"]["weight_decay"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        **(
            {"eval_strategy": cfg["training"]["eval_strategy"]}
            if "eval_strategy" in TrainingArguments.__init__.__code__.co_varnames
            else {"evaluation_strategy": cfg["training"]["eval_strategy"]}
        ),
        save_strategy=cfg["training"]["save_strategy"],
        save_total_limit=cfg["training"].get("save_total_limit", 3),
        logging_steps=cfg["training"]["logging_steps"],
        logging_first_step=True,
        disable_tqdm=False,
        fp16=cfg["hardware"]["fp16"],
        dataloader_num_workers=cfg["training"].get("dataloader_num_workers", 4),
        dataloader_pin_memory=cfg["hardware"]["pin_memory"],
        seed=cfg["training"]["seed"],
        remove_unused_columns=False,
        report_to="none",
        eval_accumulation_steps=5,
        lr_scheduler_type="polynomial",
        warmup_ratio=cfg["training"].get("warmup_ratio", 0.1),
        load_best_model_at_end=True,
        metric_for_best_model="mIoU",
        greater_is_better=True,
        ddp_find_unused_parameters=False,
    )

    best_model_dir = cfg["paths"].get(
        "best_model_dir",
        os.path.join(cfg["paths"]["output_dir"], "best_model"),
    )

    callbacks = [
        SaveBestHFCallback(save_dir=best_model_dir, processor=processor),
        EarlyStoppingCallback(
            early_stopping_patience=cfg["training"].get("early_stopping_patience", 5),
        ),
    ]

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    output_dir = cfg["paths"]["output_dir"]

    if os.path.isdir(output_dir) and any("checkpoint" in x for x in os.listdir(output_dir)):
        print("\n✅ Found checkpoint — Resuming training...\n")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("\n🚀 No checkpoint found — Starting fresh training...\n")
        trainer.train()

    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final eval: {eval_results}")

    print("Computing per-class IoU...")
    eval_output = trainer.predict(val_dataset)

    predictions = eval_output.predictions.astype(np.int64)
    labels = eval_output.label_ids

    hist = np.zeros((cfg["model"]["num_classes"], cfg["model"]["num_classes"]), dtype=np.float32)
    for p, l in zip(predictions, labels):
        hist += fast_hist(
            l.flatten(),
            p.flatten(),
            cfg["model"]["num_classes"],
        )

    per_class_ious, per_class_safe_ious = per_class_iu(
        hist,
        class_mapping,
        IMP_CLASSES,
        ind_of_imp_classes,
        num_classes=len(L3_CLASSES),
    )

    # Print per-class IoU table
    table_lines = [
        "| idx | class | IoU | Safe_IoU |",
        "| --- | ----- | --- | -------- |",
    ]
    for idx, name in enumerate(L3_CLASSES):
        iou = float(per_class_ious[idx])
        safe_iou = float(per_class_safe_ious[idx])
        table_lines.append(f"| {idx} | {name} | {iou:.4f} | {safe_iou:.4f} |")

    print("\nPer-class IoU (validation):")
    print("\n".join(table_lines))

    # Save per-class IoU to CSV
    csv_path = os.path.join(cfg["paths"]["output_dir"], "per_class_iou.csv")
    with open(csv_path, "w") as f:
        f.write("idx,class,IoU,Safe_IoU\n")
        for idx, name in enumerate(L3_CLASSES):
            iou = float(per_class_ious[idx])
            safe_iou = float(per_class_safe_ious[idx])
            f.write(f"{idx},{name},{iou:.6f},{safe_iou:.6f}\n")

    plot_dir = os.path.join(cfg["paths"]["output_dir"], "plots")

    plot_all(
        log_history=trainer.state.log_history,
        plot_dir=plot_dir,
        per_class_ious=per_class_ious,
        per_class_safe_ious=per_class_safe_ious,
        class_names=L3_CLASSES,
    )

    print("Training complete.")

if __name__ == "__main__":
    main()
