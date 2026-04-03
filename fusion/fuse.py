"""
Fuse RGB + NIR images using PIAFusion or SeAFusion.

Usage:
    python fusion/fuse.py \
        --arch piafusion \
        --weights fusion/pretrained_weights/piafusion.pth \
        --rgb_dir  d:/Mini-Project-2/data-backups/IDDAW_ICPR/val/rgb \
        --nir_dir  d:/Mini-Project-2/data-backups/IDDAW_ICPR/val/nir \
        --out_dir  d:/Mini-Project-2/data-backups/PIAFusion_ICPR/val/rgb
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# ── colour helpers ──────────────────────────────────────────────────────

def clamp(x, lo=0.0, hi=1.0):
    return torch.clamp(x, lo, hi)

def RGB2YCrCb(rgb):
    R, G, B = rgb[0:1], rgb[1:2], rgb[2:3]
    Y  = 0.299*R + 0.587*G + 0.114*B
    Cr = (R - Y)*0.713 + 0.5
    Cb = (B - Y)*0.564 + 0.5
    return clamp(Y), clamp(Cb), clamp(Cr)

def YCrCb2RGB(Y, Cb, Cr):
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, H, W = ycrcb.shape
    flat = ycrcb.reshape(3, -1).T
    mat  = torch.tensor([[1.0,1.0,1.0],[1.403,-0.714,0.0],[0.0,-0.344,1.773]], device=Y.device)
    bias = torch.tensor([0.0, -0.5, -0.5], device=Y.device)
    out  = (flat + bias).mm(mat).T.reshape(C, H, W)
    return clamp(out)


# ═══════════════════════════════════════════════════════════════════════
# PIAFusion  (keys: encoder.vi_conv1.*, encoder.vi_conv2.conv.1.*, ...,
#                    decoder.conv1.conv.1.*, ..., decoder.conv5.*)
# Copied verbatim from PIAFusion_pytorch/models/ to match state_dict.
# ═══════════════════════════════════════════════════════════════════════

class _reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0),
        )
    def forward(self, x):
        return self.conv(x)

def _cmdaf(vi, ir):
    sig = nn.Sigmoid(); gap = nn.AdaptiveAvgPool2d(1)
    d_vi_ir = vi - ir;  vi = vi + (ir - vi) * sig(gap(ir - vi))
    ir = ir + d_vi_ir * sig(gap(d_vi_ir))
    return vi, ir

class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vi_conv1 = nn.Conv2d(1, 16, 1, 1, 0)
        self.ir_conv1 = nn.Conv2d(1, 16, 1, 1, 0)
        self.vi_conv2 = _reflect_conv(16, 16, 3, 1, 1); self.ir_conv2 = _reflect_conv(16, 16, 3, 1, 1)
        self.vi_conv3 = _reflect_conv(16, 32, 3, 1, 1); self.ir_conv3 = _reflect_conv(16, 32, 3, 1, 1)
        self.vi_conv4 = _reflect_conv(32, 64, 3, 1, 1); self.ir_conv4 = _reflect_conv(32, 64, 3, 1, 1)
        self.vi_conv5 = _reflect_conv(64,128, 3, 1, 1); self.ir_conv5 = _reflect_conv(64,128, 3, 1, 1)

    def forward(self, y_vi, ir):
        act = nn.LeakyReLU()
        v = act(self.vi_conv1(y_vi)); i = act(self.ir_conv1(ir))
        v, i = _cmdaf(act(self.vi_conv2(v)), act(self.ir_conv2(i)))
        v, i = _cmdaf(act(self.vi_conv3(v)), act(self.ir_conv3(i)))
        v, i = _cmdaf(act(self.vi_conv4(v)), act(self.ir_conv4(i)))
        v, i = act(self.vi_conv5(v)), act(self.ir_conv5(i))
        return v, i

class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _reflect_conv(256, 256, 3, 1, 1)
        self.conv2 = _reflect_conv(256, 128, 3, 1, 1)
        self.conv3 = _reflect_conv(128,  64, 3, 1, 1)
        self.conv4 = _reflect_conv( 64,  32, 3, 1, 1)
        self.conv5 = nn.Conv2d(32, 1, 1, 1, 0)

    def forward(self, x):
        act = nn.LeakyReLU()
        x = act(self.conv1(x)); x = act(self.conv2(x))
        x = act(self.conv3(x)); x = act(self.conv4(x))
        return nn.Tanh()(self.conv5(x)) / 2 + 0.5

class PIAFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()

    def forward(self, y_vi, ir):
        v, i = self.encoder(y_vi, ir)
        return self.decoder(torch.cat([v, i], dim=1))


# ═══════════════════════════════════════════════════════════════════════
# SeAFusion  (keys: vis_conv.conv.*, vis_rgbd1.dense.conv1.conv.*,
#                   decode4.conv.*, decode1.conv.*, ...)
# Copied verbatim from SeAFusion/FusionNet.py to match state_dict.
# ═══════════════════════════════════════════════════════════════════════

class _ConvBnLeakyRelu2d(nn.Module):
    def __init__(self, inc, outc, ks=3, p=1, s=1, d=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, ks, s, p, d, g)
        self.bn = nn.BatchNorm2d(outc)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), 0.2)

class _ConvBnTanh2d(nn.Module):
    def __init__(self, inc, outc, ks=3, p=1, s=1, d=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, ks, s, p, d, g)
        self.bn = nn.BatchNorm2d(outc)
    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5

class _ConvLeakyRelu2d(nn.Module):
    def __init__(self, inc, outc, ks=3, p=1, s=1, d=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, ks, s, p, d, g)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), 0.2)

class _Sobelxy(nn.Module):
    def __init__(self, ch, ks=3, p=1, s=1, d=1, g=1):
        super().__init__()
        sf = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
        self.convx = nn.Conv2d(ch, ch, ks, s, p, d, ch, False)
        self.convx.weight.data.copy_(torch.from_numpy(sf))
        self.convy = nn.Conv2d(ch, ch, ks, s, p, d, ch, False)
        self.convy.weight.data.copy_(torch.from_numpy(sf.T))
    def forward(self, x):
        return torch.abs(self.convx(x)) + torch.abs(self.convy(x))

class _Conv1(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, 1, 1, 0)
    def forward(self, x):
        return self.conv(x)

class _DenseBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = _ConvLeakyRelu2d(ch, ch)
        self.conv2 = _ConvLeakyRelu2d(2*ch, ch)
    def forward(self, x):
        x = torch.cat([x, self.conv1(x)], 1)
        x = torch.cat([x, self.conv2(x)], 1)
        return x

class _RGBD(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.dense = _DenseBlock(inc)
        self.convdown = _Conv1(3*inc, outc)
        self.sobelconv = _Sobelxy(inc)
        self.convup = _Conv1(inc, outc)
    def forward(self, x):
        return F.leaky_relu(self.convdown(self.dense(x)) + self.convup(self.sobelconv(x)), 0.1)

class SeAFusion(nn.Module):
    def __init__(self):
        super().__init__()
        vc, ic = [16,32,48], [16,32,48]
        self.vis_conv  = _ConvLeakyRelu2d(1, vc[0])
        self.vis_rgbd1 = _RGBD(vc[0], vc[1])
        self.vis_rgbd2 = _RGBD(vc[1], vc[2])
        self.inf_conv  = _ConvLeakyRelu2d(1, ic[0])
        self.inf_rgbd1 = _RGBD(ic[0], ic[1])
        self.inf_rgbd2 = _RGBD(ic[1], ic[2])
        self.decode4 = _ConvBnLeakyRelu2d(vc[2]+ic[2], vc[1]+ic[1])
        self.decode3 = _ConvBnLeakyRelu2d(vc[1]+ic[1], vc[0]+ic[0])
        self.decode2 = _ConvBnLeakyRelu2d(vc[0]+ic[0], vc[0])
        self.decode1 = _ConvBnTanh2d(vc[0], 1)

    def forward(self, y_vi, ir):
        v = self.vis_rgbd2(self.vis_rgbd1(self.vis_conv(y_vi)))
        i = self.inf_rgbd2(self.inf_rgbd1(self.inf_conv(ir)))
        x = self.decode4(torch.cat([v, i], 1))
        return self.decode1(self.decode2(self.decode3(x)))


# ── inference ───────────────────────────────────────────────────────────

MODELS = {"piafusion": PIAFusion, "seafusion": SeAFusion}

def fuse_image(model, rgb_img, nir_img, device, resize=None):
    orig_size = rgb_img.size  # (W, H)
    to_tensor = transforms.ToTensor()

    if resize:
        rgb_img = rgb_img.resize(resize, Image.BILINEAR)
        nir_img = nir_img.resize(resize, Image.BILINEAR)

    rgb_t = to_tensor(rgb_img.convert("RGB"))
    nir_t = to_tensor(nir_img.convert("L"))
    y, cb, cr = RGB2YCrCb(rgb_t)
    with torch.no_grad():
        fy = model(y.unsqueeze(0).to(device), nir_t.unsqueeze(0).to(device))
        fy = clamp(fy).squeeze(0).cpu()
    fused = transforms.ToPILImage()(YCrCb2RGB(fy, cb, cr))

    if resize:
        fused = fused.resize(orig_size, Image.BILINEAR)
    return fused


def run(arch, weights, rgb_dir, nir_dir, out_dir, device="cuda", resize=None):
    model = MODELS[arch]()
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.to(device).eval()
    print(f"Loaded {arch} from {weights}")
    if resize:
        print(f"Resizing to {resize[0]}x{resize[1]} for fusion (output restored to original size)")

    pairs = []
    skipped = 0
    for root, _, files in os.walk(rgb_dir):
        for f in sorted(files):
            if not f.endswith(".png"):
                continue
            rgb_path = os.path.join(root, f)
            rel = os.path.relpath(rgb_path, rgb_dir)
            nir_path = os.path.join(nir_dir, rel.replace("_rgb.png", "_nir.png"))
            out_path = os.path.join(out_dir, rel)
            if not os.path.exists(nir_path):
                continue
            if os.path.exists(out_path):
                skipped += 1
                continue
            pairs.append((rgb_path, nir_path, rel))

    print(f"Found {len(pairs)} remaining pairs ({skipped} already done, skipped)")
    os.makedirs(out_dir, exist_ok=True)

    for rgb_path, nir_path, rel in tqdm(pairs, desc="Fusing"):
        out_path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fused = fuse_image(model, Image.open(rgb_path), Image.open(nir_path), device, resize)
        fused.save(out_path)

    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=list(MODELS.keys()))
    p.add_argument("--weights", required=True, help="path to .pth file")
    p.add_argument("--rgb_dir", required=True)
    p.add_argument("--nir_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("W", "H"),
                   help="resize before fusion to fit in GPU memory, e.g. --resize 1024 768")
    args = p.parse_args()
    resize = tuple(args.resize) if args.resize else None
    run(args.arch, args.weights, args.rgb_dir, args.nir_dir, args.out_dir, args.device, resize)
