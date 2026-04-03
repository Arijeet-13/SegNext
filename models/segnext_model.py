import os
import torch
import torch.nn as nn
from transformers import SegformerImageProcessor

from models.SegNext.model import SegNext

class SegNextWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        config = {
            # Decoder + Hamburger parameters (aligned with SegNext-Large defaults)
            # Upstream mmseg config uses channels=1024 and ham_channels=1024 for large.
            "ham_channels": 1024,
            "SPATIAL": True,
            "MD_S": 1,
            # Keep the decomposition channel size consistent with ham_channels for stability.
            "MD_D": 1024,
            "MD_R": 64,
            "TRAIN_STEPS": 6,
            "EVAL_STEPS": 6,
            "INV_T": 1,
            "BETA": 0.1,
            "Eta": 0.9,
            "RAND_INIT": True,
            "put_cheese": True,
        }

        # SegNext Large configurations
        embed_dims = [64, 128, 320, 512]
        depths = [3, 5, 27, 3]
        dec_outChannels = 1024
        
        self.model = SegNext(
            num_classes=num_classes,
            in_channnels=3,
            embed_dims=embed_dims,
            ffn_ratios=[4, 4, 4, 4],
            depths=depths,
            num_stages=4,
            dec_outChannels=dec_outChannels,
            config=config,
            dropout=0.1,
            drop_path=0.3
        )

    def forward(self, pixel_values, labels=None, **kwargs):
        logits = self.model(pixel_values)
        
        # HuggingFace expects an object with `.logits`
        class Output:
            pass
            
        output = Output()
        output.logits = logits
        
        return output

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

def load_segnext_large(num_classes: int, image_size: int = 1024, checkpoint: str = None):
    print(f"Loading SegNext-Large model wrapper")
    model = SegNextWrapper(num_classes=num_classes)

    # Optional: load finetune/pretrain checkpoint saved by SaveBestHFCallback
    if checkpoint:
        ckpt_path = checkpoint
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "pytorch_model.bin")
        if os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[SegNext] Loaded with missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print(f"[SegNext] Checkpoint not found at: {checkpoint}")
    
    # We can use SegformerImageProcessor since it applies standard ImageNet normalization
    # which is likely what the pretrained MSCANet expects as well.
    try:
        processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
            size={"height": image_size, "width": image_size} if isinstance(image_size, int) else image_size,
        )
    except Exception as e:
        processor = SegformerImageProcessor(
            size={"height": image_size, "width": image_size} if isinstance(image_size, int) else image_size,
        )

    return model, processor
