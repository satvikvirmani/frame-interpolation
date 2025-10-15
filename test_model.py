import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torch.nn.functional import interpolate as F_interpolate
from tqdm import tqdm

from classes.model import FrameInterpolationModel
from classes.dataset import FrameInterpolationDataset

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def evaluate_model(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss, total_psnr = 0.0, 0.0

    with torch.no_grad():
        loop = tqdm(dataloader, total=len(dataloader))
            
        for frame1, frame3, target in loop:
            frame1, frame3, target = frame1.to(device), frame3.to(device), target.to(device)
            inputs = torch.cat((frame1, frame3), dim=1)
            outputs = model(inputs)
            target_resized = F_interpolate(target, size=outputs.shape[2:], mode='bilinear', align_corners=True)

            loss = criterion(outputs, target_resized)
            total_loss += loss.item()
            total_psnr += psnr(outputs, target_resized).item()

    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / len(dataloader)
    return avg_loss, avg_psnr


# ---------------------------------------------
# Visualization
# ---------------------------------------------
# def show_predictions(model, dataset, device, num_samples=5):
#     model.eval()
#     indices = random.sample(range(len(dataset)), num_samples)
#     plt.figure(figsize=(15, num_samples * 3))

#     for i, idx in enumerate(indices):
#         frame1, frame3, target = dataset[idx]
#         inputs = torch.cat((frame1.unsqueeze(0), frame3.unsqueeze(0)), dim=1).to(device)
#         with torch.no_grad():
#             pred = model(inputs).cpu().squeeze(0).clamp(0, 1)

#         grid = make_grid([frame1, pred, frame3], nrow=3)
#         npimg = grid.permute(1, 2, 0).numpy()
#         plt.subplot(num_samples, 1, i + 1)
#         plt.imshow(npimg)
#         plt.axis('off')
#         plt.title(f"Sample {i+1}: Left - im1 | Middle - Predicted | Right - im3")

#     plt.tight_layout()
#     plt.show()

def show_predictions(model, dataset, device, num_samples=5):
    import torch.nn.functional as F
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    plt.figure(figsize=(15, num_samples * 3))

    for i, idx in enumerate(indices):
        frame1, frame3, target = dataset[idx]
        inputs = torch.cat((frame1.unsqueeze(0), frame3.unsqueeze(0)), dim=1).to(device)
        with torch.no_grad():
            pred = model(inputs).cpu().squeeze(0).clamp(0, 1)

        # --- Resize all tensors to same size for display ---
        H = min(frame1.shape[1], frame3.shape[1], pred.shape[1])
        W = min(frame1.shape[2], frame3.shape[2], pred.shape[2])
        frame1 = F.interpolate(frame1.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=True).squeeze(0)
        frame3 = F.interpolate(frame3.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=True).squeeze(0)
        pred   = F.interpolate(pred.unsqueeze(0),   size=(H, W), mode='bilinear', align_corners=True).squeeze(0)

        # --- Make grid and display ---
        grid = make_grid([frame1, pred, frame3], nrow=3)
        npimg = grid.permute(1, 2, 0).numpy()
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(npimg)
        plt.axis('off')
        plt.title(f"Sample {i+1}: Left - im1 | Middle - Predicted | Right - im3")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------
# Main Script
# ---------------------------------------------
if __name__ == "__main__":
    root_dir = 'vimeo_triplet'
    model_path = 'models/frame_interpolation.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("Loading test dataset...")
    test_dataset = FrameInterpolationDataset(root_dir, 'tri_testlist.txt', transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    print(f"Loaded {len(test_dataset)} test samples.")

    # Load model
    model = FrameInterpolationModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Model loaded from {model_path}")

    # Evaluate
    print("\nEvaluating model on test set...")
    test_loss, test_psnr = evaluate_model(model, test_loader, device)
    print(f"📊 Test MSE Loss: {test_loss:.6f}")
    print(f"📈 Test PSNR: {test_psnr:.2f} dB")

    # Visualize 5 random predictions
    print("\n🎨 Displaying 5 random predictions...")
    show_predictions(model, test_dataset, device, num_samples=5)