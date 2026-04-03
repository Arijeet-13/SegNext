import torch
import torch.nn as nn
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers import SegformerImageProcessor

from model import SegNext

class SegNextForHF(nn.Module):
    def __init__(self, num_classes, pretrained_path=None, config=None):
        super().__init__()
        # SegNeXt-Large config estimate
        self.segnext = SegNext(
            num_classes=num_classes,
            in_channnels=3,
            embed_dims=[64, 128, 320, 512],
            ffn_ratios=[8, 8, 4, 4],
            depths=[3, 5, 27, 3],
            num_stages=4,
            dec_outChannels=1024, # Adjust if needed
            config=config,
            drop_path=0.2
        )
        self.num_labels = num_classes
        
        if pretrained_path is not None:
             print(f"Loading weights from {pretrained_path}")
             try:
                 checkpoint = torch.load(pretrained_path, map_location='cpu')
                 # Often weights are under 'state_dict' or 'model'
                 state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
                 # Remove 'module.' prefix if saved via DataParallel
                 state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                 # strict=False in case classifier head dimensions mismatch (e.g. Cityscapes has 19 classes, IDD AW has 26)
                 self.segnext.load_state_dict(state_dict, strict=False)
             except Exception as e:
                 print(f"Warning: Failed to load pretrained weights from {pretrained_path} -> {e}")

    def forward(
        self,
        pixel_values=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else True

        logits = self.segnext(pixel_values)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=255)
            # Resize logits to match labels size if they don't match
            if logits.shape[-2:] != labels.shape[-2:]:
                logits_resized = torch.nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
            else:
                logits_resized = logits
            loss = loss_fct(logits_resized, labels)

        if not return_dict:
            return tuple(v for v in [loss, logits] if v is not None)

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
        )

    def save_pretrained(self, save_dir):
        import os
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

def load_segnext_large(num_classes, pretrained_path=None, image_size=768, hamburger_cfg=None):
    model = SegNextForHF(num_classes=num_classes, pretrained_path=pretrained_path, config=hamburger_cfg)
    # Using Segformer image processor as a generic ImageNet normalizer
    processor = SegformerImageProcessor(
        size={"height": image_size, "width": image_size} if isinstance(image_size, int) else image_size,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        do_resize=True,
        do_rescale=True
    )
    return model, processor
