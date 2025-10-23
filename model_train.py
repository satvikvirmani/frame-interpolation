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
#from classes.cnn import FrameInterpolationModel
from classes.dataset import FrameInterpolationDataset
from classes.loss import SSIM_MSE_Loss
from classes.lr import cosine_lr

CHECKPOINT_PATH = "checkpoints/checkpoint.pth"
FINAL_MODEL_PATH = "checkpoints/frame_interpolation.pth"
DATASET_ROOT_PATH = "vimeo_triplet"
EPOCHS = 10
BATCH_SIZE = 32
INITIAL_LR = 1e-3
PIN_MEMORY = True
NUM_WORKERS = 4
# Experimental: enable mixed-precision on MPS (might be unstable / unsupported for some ops).
# Default is False because GradScaler support on MPS is still flaky in some PyTorch versions.
USE_MPS_AMP = False

os.makedirs("checkpoints", exist_ok=True)

def get_device():
    return torch.device("cpu")
    # Prioritized: MPS (Apple Metal) -> CUDA -> CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS (Apple GPU) selected")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device: CUDA (NVIDIA GPU) selected")
    else:
        device = torch.device("cpu")
        print("Device: CPU selected")
    return device

def to_device(batch, device):
    # Move tensors in the batch to device. Works for tuples/lists/dicts of tensors.
    if isinstance(batch, (tuple, list)):
        return [to_device(x, device) for x in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch.to(device, non_blocking=True)

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
        device = get_device()

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        print("Loading Vimeo-90K dataset...")
        train_dataset = FrameInterpolationDataset(DATASET_ROOT_PATH, 'tri_trainlist.txt', transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY  if device.type != "mps" else False)
        print(f"Loaded {len(train_dataset)} training samples")

        model = FrameInterpolationModel().to(device)
        criterion = SSIM_MSE_Loss(alpha=0.85)
        optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-5)
        
        use_amp = False
        scaler = None
        if device.type == "cuda":
            use_amp = True
            scaler = GradScaler()  # safe & recommended on CUDA
            print("AMP enabled with GradScaler for CUDA.")
        elif device.type == "mps":
            # Option: If you want to experiment with autocast on MPS (PyTorch >= 2.5),
            # you can set USE_MPS_AMP = True. Be aware of known issues with GradScaler on MPS.
            if USE_MPS_AMP:
                # we will use autocast for mps but NOT GradScaler (scaling may raise errors).
                use_amp = True
                scaler = None
                print("Experimental: AMPCast enabled for MPS (without GradScaler).")
            else:
                use_amp = False
                scaler = None
                print("AMP disabled for MPS. Training in full float32 for stability.")
        else:
            use_amp = False
            scaler = None
            print("AMP disabled on CPU by default.")

        start_epoch = load_checkpoint(model, optimizer)

        print("\nStarting training...\n")
        
        for epoch in range(start_epoch, EPOCHS + 1):
            epoch_start = time.time()
            running_loss = 0.0
            
            for g in optimizer.param_groups:
                g['lr'] = cosine_lr(epoch, EPOCHS, INITIAL_LR)
            
            pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch}/{EPOCHS}]")

            for batch in pbar:
                try:
                    frame1, frame2, target = batch
                    inputs = torch.cat((frame1, frame2), dim=1)

                    inputs = inputs.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    
                    if use_amp and device.type == "cuda":
                        # Standard CUDA AMP with GradScaler
                        with autocast(device_type="cuda"):
                            outputs = model(inputs)
                            target_resized = F_interpolate(target, size=outputs.shape[2:], mode='bilinear', align_corners=True)
                            loss = criterion(outputs, target_resized)
                        # scale + backprop + step
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    elif use_amp and device.type == "mps":
                        # Experimental: autocast for mps but without GradScaler.
                        # This may speed up some ops, but some ops might not have mps kernels.
                        # If you hit errors, set USE_MPS_AMP = False.
                        try:
                            with torch.amp.autocast(device_type="mps"):
                                outputs = model(inputs)
                                loss = criterion(outputs, target)
                            # backprop normally (no scaler)
                            loss.backward()
                            optimizer.step()
                        except Exception as e_mps:
                            # On rare unsupported-op errors, fallback to CPU for this batch:
                            print("MPS runtime error in autocast step:", e_mps)
                            print("Falling back to CPU for this batch...")
                            inputs_cpu = inputs.cpu()
                            target_cpu = target.cpu()
                            outputs_cpu = model.to("cpu")(inputs_cpu)  # move model to cpu temporarily
                            loss_cpu = criterion(outputs_cpu, target_cpu)
                            loss_cpu.backward()
                            optimizer.step()
                            model.to(device)  # move it back to original device
                            loss = loss_cpu.to(device)
                    else:
                        # No AMP (full float32)
                        outputs = model(inputs)
                        loss = criterion(outputs, target)
                        loss.backward()
                        optimizer.step()
                        

                    batch_loss = float(loss.detach().cpu().item())
                    running_loss += batch_loss
                    pbar.set_postfix({'loss': f'{batch_loss:.5f}'})
                except RuntimeError as e:
                    # Generic runtime error: try fallback to CPU for this batch
                    print(f"RuntimeError during training batch: {e}")
                    print("Falling back to CPU for this batch...")
                    try:
                        frame1_cpu, frame2_cpu, target_cpu = [t.cpu() for t in (frame1, frame2, target)]
                        inputs_cpu = torch.cat((frame1_cpu, frame2_cpu), dim=1)
                        outputs_cpu = model.to("cpu")(inputs_cpu)
                        loss_cpu = criterion(outputs_cpu, target_cpu)
                        loss_cpu.backward()
                        optimizer.step()
                        model.to(device)
                        batch_loss = float(loss_cpu.detach().cpu().item())
                        epoch_loss += batch_loss
                    except Exception as e2:
                        print("Failed fallback on CPU:", e2)
                        # skip this batch
                        continue

            avg_loss = running_loss / max(1, len(train_loader))
            epoch_time = time.time() - epoch_start
            print(f"Epoch [{epoch+1}/{EPOCHS}] - LR: {optimizer.param_groups[0]['lr']:.6f} - Loss: {avg_loss:.6f} - Time {epoch_time:.1f}s")
                        
            save_checkpoint(model, optimizer, epoch, avg_loss)
            
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        print("\nTraining complete. Model saved to:", FINAL_MODEL_PATH)