"""
Fusion quality metrics from:
  "A review on infrared and visible image fusion algorithms based on neural networks"
  Yang et al., J. Vis. Commun. Image R. 101 (2024) 104179

All 11 metrics from Section 3.2:
  EN, MI, SF, AG, SD, SSIM, CC, PSNR, VIF, Qabf, SCD

Every function takes numpy arrays (grayscale, float64, range [0,1]).
For colour images, convert to grayscale before calling.

Usage:
    python fusion/metrics.py \
        --fused_dir  data/PIAFusion_fused/val/rgb \
        --rgb_dir    data/IDDAW_ICPR/val/rgb \
        --nir_dir    data/IDDAW_ICPR/val/nir
"""
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import convolve2d


# ── helpers ─────────────────────────────────────────────────────────────

def _to_gray(img):
    """Convert HWC uint8/float image to float64 grayscale [0,1]."""
    if img.ndim == 3:
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def _joint_hist(a, b, bins=256):
    """Joint histogram of two images."""
    a_q = np.clip((a * (bins - 1)).astype(int), 0, bins - 1)
    b_q = np.clip((b * (bins - 1)).astype(int), 0, bins - 1)
    hist = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(hist, (a_q.ravel(), b_q.ravel()), 1)
    hist /= hist.sum()
    return hist


# ── 1. Entropy (EN) ────────────────────────────────────────────────────

def entropy(fused):
    """Information richness of fused image."""
    hist, _ = np.histogram(fused, bins=256, range=(0, 1), density=True)
    hist = hist[hist > 0] * (1.0 / 256)  # normalize to probabilities
    # recompute properly
    counts, _ = np.histogram(fused, bins=256, range=(0, 1))
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


# ── 2. Mutual Information (MI) ─────────────────────────────────────────

def mutual_information(src, fused, bins=256):
    """MI between one source image and the fused image."""
    joint = _joint_hist(src, fused, bins)
    p_src = joint.sum(axis=1)
    p_fus = joint.sum(axis=0)

    # MI = sum p(x,f) * log2(p(x,f) / (p(x)*p(f)))
    mask = joint > 0
    mi = np.sum(
        joint[mask] * np.log2(joint[mask] / (p_src[np.newaxis, :].T * p_fus[np.newaxis, :])[mask])
    )
    return mi


def MI(rgb, nir, fused):
    """MI = MI(rgb, fused) + MI(nir, fused)"""
    return mutual_information(rgb, fused) + mutual_information(nir, fused)


# ── 3. Spatial Frequency (SF) ──────────────────────────────────────────

def spatial_frequency(fused):
    """Gradient-based sharpness metric."""
    M, N = fused.shape
    rf = np.sqrt(np.sum((fused[:, 1:] - fused[:, :-1]) ** 2) / (M * N))
    cf = np.sqrt(np.sum((fused[1:, :] - fused[:-1, :]) ** 2) / (M * N))
    return np.sqrt(rf ** 2 + cf ** 2)


# ── 4. Average Gradient (AG) ──────────────────────────────────────────

def average_gradient(fused):
    """Texture detail richness."""
    dx = fused[:-1, :-1] - fused[1:, :-1]
    dy = fused[:-1, :-1] - fused[:-1, 1:]
    M, N = fused.shape
    return np.mean(np.sqrt((dx ** 2 + dy ** 2) / 2.0))


# ── 5. Standard Deviation (SD) ─────────────────────────────────────────

def standard_deviation(fused):
    """Contrast / gray-level spread."""
    return np.std(fused)


# ── 6. SSIM ─────────────────────────────────────────────────────────────

def _ssim_single(src, fused, win_size=11, C1=0.01**2, C2=0.03**2):
    """SSIM between one source and fused image."""
    pad = win_size // 2
    mu_x = uniform_filter(src, size=win_size)
    mu_f = uniform_filter(fused, size=win_size)
    sigma_x2 = uniform_filter(src ** 2, size=win_size) - mu_x ** 2
    sigma_f2 = uniform_filter(fused ** 2, size=win_size) - mu_f ** 2
    sigma_xf = uniform_filter(src * fused, size=win_size) - mu_x * mu_f

    num = (2 * mu_x * mu_f + C1) * (2 * sigma_xf + C2)
    den = (mu_x ** 2 + mu_f ** 2 + C1) * (sigma_x2 + sigma_f2 + C2)
    ssim_map = num / den

    # crop borders
    return ssim_map[pad:-pad, pad:-pad].mean()


def SSIM(rgb, nir, fused, w_rgb=0.5, w_nir=0.5):
    """Weighted SSIM: w_rgb * SSIM(rgb, fused) + w_nir * SSIM(nir, fused)."""
    return w_rgb * _ssim_single(rgb, fused) + w_nir * _ssim_single(nir, fused)


# ── 7. Correlation Coefficient (CC) ────────────────────────────────────

def _cc(src, fused):
    """Pearson correlation between two images."""
    x = src - src.mean()
    f = fused - fused.mean()
    num = np.sum(x * f)
    den = np.sqrt(np.sum(x ** 2) * np.sum(f ** 2))
    return num / den if den > 0 else 0.0


def CC(rgb, nir, fused, w_rgb=0.5, w_nir=0.5):
    """Weighted CC."""
    return w_rgb * _cc(rgb, fused) + w_nir * _cc(nir, fused)


# ── 8. PSNR ─────────────────────────────────────────────────────────────

def _mse(src, fused):
    return np.mean((src - fused) ** 2)


def PSNR(rgb, nir, fused, peak=1.0, w_rgb=0.5, w_nir=0.5):
    """Weighted PSNR. Images in [0,1] so peak=1.0."""
    mse = w_rgb * _mse(rgb, fused) + w_nir * _mse(nir, fused)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(peak ** 2 / mse)


# ── 9. VIF (Visual Information Fidelity) ────────────────────────────────

def _vif_single(src, fused, sigma_nsq=2.0):
    """
    Simplified VIF between one source and fused image.
    Uses a multi-scale (4-level) Gaussian downsampling approach.
    """
    eps = 1e-10
    num = 0.0
    den = 0.0

    for scale in range(4):
        if scale > 0:
            # Gaussian downsample
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
            src = convolve2d(src, kernel, mode="same", boundary="symm")[::2, ::2]
            fused = convolve2d(fused, kernel, mode="same", boundary="symm")[::2, ::2]

        mu_s = uniform_filter(src, size=3)
        mu_f = uniform_filter(fused, size=3)
        sigma_s2 = uniform_filter(src ** 2, size=3) - mu_s ** 2
        sigma_f2 = uniform_filter(fused ** 2, size=3) - mu_f ** 2
        sigma_sf = uniform_filter(src * fused, size=3) - mu_s * mu_f

        sigma_s2 = np.maximum(sigma_s2, 0)
        sigma_f2 = np.maximum(sigma_f2, 0)

        g = sigma_sf / (sigma_s2 + eps)
        sv = sigma_f2 - g * sigma_sf

        g[sigma_s2 < eps] = 0
        sv[sigma_s2 < eps] = sigma_f2[sigma_s2 < eps]
        sigma_s2[sigma_s2 < eps] = 0
        sv[sv < eps] = eps

        num += np.sum(np.log2(1 + g ** 2 * sigma_s2 / (sv + sigma_nsq)))
        den += np.sum(np.log2(1 + sigma_s2 / sigma_nsq))

    return num / den if den > 0 else 0.0


def VIF(rgb, nir, fused, w_rgb=0.5, w_nir=0.5):
    """Weighted VIF."""
    return w_rgb * _vif_single(rgb, fused) + w_nir * _vif_single(nir, fused)


# ── 10. Qabf ────────────────────────────────────────────────────────────

def _sobel_strength(img):
    """Gradient strength via Sobel."""
    sx = convolve2d(img, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64),
                    mode="same", boundary="symm")
    sy = convolve2d(img, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64),
                    mode="same", boundary="symm")
    return np.sqrt(sx ** 2 + sy ** 2), np.arctan2(sy, sx)


def _q0(src, fused, win_size=8):
    """Edge-preserving quality measure Q0 between src and fused."""
    mu_s = uniform_filter(src, size=win_size)
    mu_f = uniform_filter(fused, size=win_size)
    sigma_s2 = uniform_filter(src ** 2, size=win_size) - mu_s ** 2
    sigma_f2 = uniform_filter(fused ** 2, size=win_size) - mu_f ** 2
    sigma_sf = uniform_filter(src * fused, size=win_size) - mu_s * mu_f

    sigma_s = np.sqrt(np.maximum(sigma_s2, 0))
    sigma_f = np.sqrt(np.maximum(sigma_f2, 0))

    eps = 1e-10
    q_corr = sigma_sf / (sigma_s * sigma_f + eps)
    q_lum = 2 * mu_s * mu_f / (mu_s ** 2 + mu_f ** 2 + eps)
    q_con = 2 * sigma_s * sigma_f / (sigma_s2 + sigma_f2 + eps)

    return q_corr * q_lum * q_con


def Qabf(rgb, nir, fused):
    """Edge-based fusion quality metric."""
    s_a, _ = _sobel_strength(rgb)
    s_b, _ = _sobel_strength(nir)

    lam = s_a / (s_a + s_b + 1e-10)  # saliency weighting

    q_af = _q0(rgb, fused)
    q_bf = _q0(nir, fused)

    return np.mean(lam * q_af + (1 - lam) * q_bf)


# ── 11. SCD (Sum of Correlations of Differences) ───────────────────────

def SCD(rgb, nir, fused):
    """Correlation between difference images and sources."""
    d1 = fused - nir  # difference w.r.t. nir -> should correlate with rgb
    d2 = fused - rgb  # difference w.r.t. rgb -> should correlate with nir
    return _cc(d1, rgb) + _cc(d2, nir)


# ── Compute all metrics at once ─────────────────────────────────────────

def compute_all(rgb, nir, fused):
    """
    Compute all 11 fusion metrics.

    Args:
        rgb:   grayscale float64 [0,1] of visible image
        nir:   grayscale float64 [0,1] of infrared image
        fused: grayscale float64 [0,1] of fused image

    Returns:
        dict with keys: EN, MI, SF, AG, SD, SSIM, CC, PSNR, VIF, Qabf, SCD
    """
    return {
        "EN":   entropy(fused),
        "MI":   MI(rgb, nir, fused),
        "SF":   spatial_frequency(fused),
        "AG":   average_gradient(fused),
        "SD":   standard_deviation(fused),
        "SSIM": SSIM(rgb, nir, fused),
        "CC":   CC(rgb, nir, fused),
        "PSNR": PSNR(rgb, nir, fused),
        "VIF":  VIF(rgb, nir, fused),
        "Qabf": Qabf(rgb, nir, fused),
        "SCD":  SCD(rgb, nir, fused),
    }


# ── CLI: batch evaluate a fused directory ───────────────────────────────

if __name__ == "__main__":
    import os
    import argparse
    from PIL import Image

    p = argparse.ArgumentParser(description="Compute fusion quality metrics")
    p.add_argument("--fused_dir", required=True, help="Directory of fused images")
    p.add_argument("--rgb_dir", required=True, help="Directory of original RGB images")
    p.add_argument("--nir_dir", required=True, help="Directory of original NIR images")
    p.add_argument("--max_images", type=int, default=None, help="Limit number of images")
    args = p.parse_args()

    # find image triplets
    triplets = []
    for root, _, files in os.walk(args.fused_dir):
        for f in sorted(files):
            if not f.endswith(".png"):
                continue
            fused_path = os.path.join(root, f)
            rel = os.path.relpath(fused_path, args.fused_dir)
            rgb_path = os.path.join(args.rgb_dir, rel)
            nir_rel = rel.replace("_rgb.png", "_nir.png")
            nir_path = os.path.join(args.nir_dir, nir_rel)
            if os.path.exists(rgb_path) and os.path.exists(nir_path):
                triplets.append((rgb_path, nir_path, fused_path))

    if args.max_images:
        triplets = triplets[: args.max_images]

    print(f"Evaluating {len(triplets)} images...")

    all_metrics = []
    from tqdm import tqdm

    for rgb_path, nir_path, fused_path in tqdm(triplets):
        rgb = _to_gray(np.array(Image.open(rgb_path)))
        nir = _to_gray(np.array(Image.open(nir_path)))
        fused = _to_gray(np.array(Image.open(fused_path)))
        all_metrics.append(compute_all(rgb, nir, fused))

    # average
    keys = all_metrics[0].keys()
    print("\n" + "=" * 50)
    print(f"Results over {len(all_metrics)} images:")
    print("=" * 50)
    for k in keys:
        vals = [m[k] for m in all_metrics]
        print(f"  {k:6s}  {np.mean(vals):8.4f}  (std {np.std(vals):.4f})")
    print("=" * 50)
