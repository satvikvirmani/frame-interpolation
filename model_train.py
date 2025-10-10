# model_train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.functional import interpolate
from PIL import Image
import time
from tqdm import tqdm

CHECKPOINT_PATH = "models/checkpoint.pth"
FINAL_MODEL_PATH = "models/frame_interpolation.pth"
EPOCHS = 10

# Create folder
os.makedirs("models", exist_ok=True)

# --- Model ---
class FrameInterpolationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(6, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(inplace=True)
            ) for _ in range(3)
        ])
        self.resize = nn.Upsample(size=(90, 160), mode='bilinear', align_corners=True)
        self.fusion_conv = nn.Conv2d(384, 128, 3, 1, 1)
        self.upsample_conv = nn.ConvTranspose2d(128, 3, 3, 2, 1, 1, 1)

    def forward(self, x):
        features = [f(x) for f in self.feature_extractor]
        resized = [self.resize(f) for f in features]
        x = torch.cat(resized, dim=1)
        x = torch.relu(self.fusion_conv(x))
        return self.upsample_conv(x)

# --- Dataset ---
class FrameInterpolationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.video_list = os.listdir(root_dir)
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.video_list) * 100

    def __getitem__(self, idx):
        v_idx, f_idx = divmod(idx, 100)
        folder = os.path.join(self.root_dir, self.video_list[v_idx])
        f1, f2, f3 = [os.path.join(folder, f"frame{f_idx+i:04d}.jpg") for i in (1,2,3)]
        imgs = [self.transform(Image.open(p)) for p in (f1, f2, f3)]
        return imgs[0], imgs[1], imgs[2]

# Function to save checkpoint
def save_checkpoint(model, optimizer, epoch, loss):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"💾 Saved checkpoint after epoch {epoch} (loss={loss:.6f})")

# Function to load checkpoint
def load_checkpoint(model, optimizer):
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔁 Resuming training from epoch {start_epoch} (previous loss={checkpoint['loss']:.6f})")
        return start_epoch
    print("🆕 No checkpoint found, starting fresh.")
    return 1

if __name__ == "__main__":
    if os.path.exists("models/frame_interpolation.pth"):
        print("Model already exists. Skipping training.")
    else:
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = FrameInterpolationModel().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        start_epoch = load_checkpoint(model, optimizer)
        
        dataset = FrameInterpolationDataset("data/frames")
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


        for epoch in range(start_epoch, EPOCHS + 1):
            epoch_start = time.time()
            running_loss = 0.0
            
            loop = tqdm(dataloader, total=len(dataloader), desc=f"Epoch [{epoch+1}/{EPOCHS}]")
            
            for frame1, frame2, target in loop:
                inputs = torch.cat((frame1, frame2), 1).to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, interpolate(target, size=outputs.shape[2:]))
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
            avg_loss = running_loss / len(dataloader)
            epoch_time = time.time() - epoch_start
            print(f"\n✅ Epoch [{epoch}/{EPOCHS}] completed — Avg Loss: {avg_loss:.6f} — Time: {epoch_time:.1f}s")
            
            save_checkpoint(model, optimizer, epoch, avg_loss)
            
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        print("\n Training complete. Model saved to:", FINAL_MODEL_PATH)