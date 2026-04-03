import torch
import torch.nn as nn
import torch.nn.functional as F
from .convnext import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge, LISA
from .decoder import MLPDecoder
from transformers.modeling_outputs import SemanticSegmenterOutput

class ConvNeXtSegmenter(nn.Module):
    """
    Complete Semantic Segmentation Model combining ConvNeXt backbone and All-MLP Decoder.
    """
    def __init__(self, backbone_type='tiny', num_classes=19, pretrained=True, use_lisa=False):
        super().__init__()
        
        # 1. Initialize Backbone
        if backbone_type == 'tiny':
            self.backbone = convnext_tiny(pretrained=pretrained)
            in_channels = [96, 192, 384, 768]
        elif backbone_type == 'small':
            self.backbone = convnext_small(pretrained=pretrained)
            in_channels = [96, 192, 384, 768]
        elif backbone_type == 'base':
            self.backbone = convnext_base(pretrained=pretrained, in_22k=True)
            in_channels = [128, 256, 512, 1024]
        elif backbone_type == 'large':
            self.backbone = convnext_large(pretrained=pretrained, in_22k=True)
            in_channels = [192, 384, 768, 1536]
        elif backbone_type == 'xlarge':
            self.backbone = convnext_xlarge(pretrained=pretrained, in_22k=True)
            in_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        # 2. Apply LISA wrapper if memory-efficient training is requested
        if use_lisa:
            # randomly unfreeze 2 intermediate blocks every 10 steps to save memory
            self.backbone = LISA(self.backbone, num_layers=2, interval=10)
            
        # 3. Initialize MLP Decoder
        self.decoder = MLPDecoder(
            in_channels=in_channels,
            embed_dim=768,       # Common embedding space for all channels
            num_classes=num_classes,
            dropout_ratio=0.1
        )

    def forward(self, pixel_values, labels=None, **kwargs):
        # Extract features from ConvNeXt encoder
        H, W = pixel_values.size(2), pixel_values.size(3)
        features = self.backbone(pixel_values, return_interm_layers=True)
        
        # Pass multi-scale features through the MLP Decoder
        out = self.decoder(features)
        
        # Hugging Face Trainer expects an object with `.logits`
        return SemanticSegmenterOutput(
            loss=None, # Loss is natively computed in CustomTrainer inside train.py
            logits=out,
        )
