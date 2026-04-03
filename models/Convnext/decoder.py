import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Linear Embedding. Uses a Linear layer to project the channel dimensions.
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # x is (N, C, H, W)
        N, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # (N, H*W, C)
        x = self.proj(x)                 # (N, H*W, embed_dim)
        x = x.transpose(1, 2).reshape(N, -1, H, W) # (N, embed_dim, H, W)
        return x

class MLPDecoder(nn.Module):
    """
    Lightweight All-MLP Decoder for Semantic Segmentation.
    Widely known from "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers".
    It unifies multi-scale features via MLPs, upsamples to a consistent resolution, and fuses them to predict masks.
    """
    def __init__(self, in_channels=[96, 192, 384, 768], embed_dim=768, num_classes=19, dropout_ratio=0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # MLP layers to unify features across all resolutions to `embed_dim`
        self.linear_c1 = MLP(input_dim=self.in_channels[0], embed_dim=self.embed_dim)
        self.linear_c2 = MLP(input_dim=self.in_channels[1], embed_dim=self.embed_dim)
        self.linear_c3 = MLP(input_dim=self.in_channels[2], embed_dim=self.embed_dim)
        self.linear_c4 = MLP(input_dim=self.in_channels[3], embed_dim=self.embed_dim)
        
        # Fusion Multi-Layer Perceptron (implemented via 1x1 Conv for 2D spatial features)
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embed_dim * 4, out_channels=self.embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final segmentation classification head
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: List of multi-scale feature maps [c1, c2, c3, c4] 
                      from backbone resolutions 1/4, 1/8, 1/16, 1/32
        """
        c1, c2, c3, c4 = features
        
        # Unify channel dimensions to `embed_dim` using MLPs
        _c1 = self.linear_c1(c1)
        _c2 = self.linear_c2(c2)
        _c3 = self.linear_c3(c3)
        _c4 = self.linear_c4(c4)
        
        # Upsample c2, c3, c4 to the spatial resolution of c1 (highest resolution map)
        target_size = _c1.shape[2:]
        _c2 = F.interpolate(_c2, size=target_size, mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=target_size, mode='bilinear', align_corners=False)
        _c4 = F.interpolate(_c4, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate across the unified embedding channels
        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1) # Shape: (N, embed_dim * 4, H, W)
        
        # Fuse channels via final MLP logic
        x = self.linear_fuse(_c) # Shape: (N, embed_dim, H, W)
        
        # Prediction
        x = self.dropout(x)
        x = self.linear_pred(x)  # Shape: (N, num_classes, H, W)
        
        return x
