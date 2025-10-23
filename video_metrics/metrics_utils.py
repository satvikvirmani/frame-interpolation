import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips


def preprocess_for_lpips(img, device):
    """Convert BGR (OpenCV) image to normalized tensor for LPIPS."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = (tensor * 2) - 1  # normalize to [-1, 1]
    return tensor.to(device)


def compute_mse(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return np.mean((img1_gray.astype("float") - img2_gray.astype("float")) ** 2)


def compute_inv_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_val, _ = ssim(img1_gray, img2_gray, full=True, data_range=255)
    return 1 - ssim_val


def compute_lpips(img1, img2, loss_fn, device):
    t1 = preprocess_for_lpips(img1, device)
    t2 = preprocess_for_lpips(img2, device)
    with torch.no_grad():
        lpips_val = loss_fn(t1, t2).item()
    return lpips_val


def combined_difference(mse_val, inv_ssim_val, lpips_val, alpha=0.5, beta=0.3, gamma=0.2):
    """Weighted composite difference metric."""
    return (alpha * mse_val) + (beta * inv_ssim_val) + (gamma * lpips_val)


def load_lpips_model(device):
    """Load LPIPS model on device."""
    return lpips.LPIPS(net='alex').to(device)