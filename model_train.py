import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.functional import interpolate as F_interpolate
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from classes.en_de import FrameInterpolationModel
from classes.dataset import FrameInterpolationDataset
from classes.loss import SSIM_MSE_Loss
from classes.lr import cosine_lr

CHECKPOINT_PATH = "checkpoints/checkpoint.pth"
FINAL_MODEL_PATH = "checkpoints/frame_interpolation.pth"
DATASET_ROOT_PATH = "vimeo_triplet"
EPOCHS = 10
BATCH_SIZE = 8
INITIAL_LR = 1e-3

os.makedirs("checkpoints", exist_ok=True)

def save_checkpoint(model, optimizer, epoch, loss):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Saved checkpoint after epoch {epoch} (loss={loss:.6f})")

def load_checkpoint(model, optimizer):
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch} (previous loss={checkpoint['loss']:.6f})")
        return start_epoch
    print("No checkpoint found, starting fresh.")
    return 1

if __name__ == "__main__":
    if os.path.exists("checkpoints/frame_interpolation.pth"):
        print("Model already exists. Skipping training.")
    else:

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        print("Loading Vimeo-90K dataset...")
        train_dataset = FrameInterpolationDataset(DATASET_ROOT_PATH, 'tri_trainlist.txt', transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        print(f"Loaded {len(train_dataset)} training samples")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = FrameInterpolationModel().to(device)
        criterion = SSIM_MSE_Loss(alpha=0.85)
        optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-5)
        scaler = GradScaler(device.type)

        start_epoch = load_checkpoint(model, optimizer)

        print("\nStarting training...\n")
        
        for epoch in range(start_epoch, EPOCHS + 1):
            epoch_start = time.time()
            running_loss = 0.0
            
            for g in optimizer.param_groups:
                g['lr'] = cosine_lr(epoch, EPOCHS, INITIAL_LR)
            
            pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch}/{EPOCHS}]")
            
            for frame1, frame3, target in pbar:                
                frame1, frame3, target = frame1.to(device), frame3.to(device), target.to(device)
                inputs = torch.cat((frame1, frame3), dim=1)

                optimizer.zero_grad()
                with autocast(device_type=device.type):
                    outputs = model(inputs)
                    target_resized = F_interpolate(target, size=outputs.shape[2:], mode='bilinear', align_corners=True)
                    loss = criterion(outputs, target_resized)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()


                running_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.5f}'})


            avg_loss = running_loss / len(train_loader)
            epoch_time = time.time() - epoch_start
            print(f"Epoch [{epoch+1}/{EPOCHS}] - LR: {optimizer.param_groups[0]['lr']:.6f} - Loss: {avg_loss:.6f} - Time {epoch_time:.1f}s")
                        
            save_checkpoint(model, optimizer, epoch, avg_loss)
            
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        print("\nTraining complete. Model saved to:", FINAL_MODEL_PATH)