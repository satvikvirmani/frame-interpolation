import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from moviepy import ImageSequenceClip
from multiframe_interpolater import MultiFrameInterpolator
from metrics_utils import compute_mse, compute_inv_ssim, compute_lpips, load_lpips_model
import warnings

# ----------------------------- #
# Global Warning Suppression
# ----------------------------- #
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TensorFlow logs suppression

# ----------------------------- #
# Video Reconstructor Class
# ----------------------------- #
class VideoReconstructor:
    def __init__(self, model, input_size=(128, 224)):
        """Initializes the reconstructor with a given interpolation model."""
        self.interpolator = MultiFrameInterpolator(model, input_size=input_size)

    # ------------------------- #
    # Video I/O
    # ------------------------- #
    def read_video(self, video_path):
        """Reads a video and returns frames + FPS."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        print(f"Video loaded: {len(frames)} frames at {fps:.2f} FPS")
        return frames, fps

    def play_video(self, frames, fps, title="preview"):
        """Writes a sequence of frames to disk as MP4."""
        valid_frames = [f for f in frames if isinstance(f, np.ndarray)]
        if not valid_frames:
            print("No valid frames to display.")
            return

        # Convert to BGR for OpenCV compatibility
        valid_frames = [cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2BGR) for f in valid_frames]
        out_path = os.path.join(os.getcwd(), f"{title}.mp4")

        clip = ImageSequenceClip(valid_frames, fps=fps)
        clip.write_videofile(out_path, codec="libx264", audio=False, logger=None)

        print(f"Saved: {out_path}")

    # ------------------------- #
    # Frame Reconstruction
    # ------------------------- #
    def reconstruct_frames(self, frames, csv_path):
        """Reconstructs missing frames using recursive interpolation."""
        df = pd.read_csv(csv_path)
        available = df['Frame_Index'].tolist()

        full_frames = [None] * len(frames)

        # Place known frames
        for idx in available:
            full_frames[idx] = frames[idx]

        # Interpolate missing ranges
        print(f"🔧 Reconstructing frames (total: {len(frames)}) ...")
        for i in tqdm(range(len(available) - 1), desc="Interpolating", ncols=90):
            s_idx, e_idx = available[i], available[i + 1]
            f_start, f_end = full_frames[s_idx], full_frames[e_idx]
            num_missing = e_idx - s_idx - 1

            if num_missing <= 0:
                continue

            depth = int(np.ceil(np.log2(num_missing + 1)))
            preds = self.recursive_interpolate(f_start, f_end, depth)

            # Adjust count
            if len(preds) > num_missing:
                step = np.linspace(0, len(preds) - 1, num_missing, dtype=int)
                preds = [preds[j] for j in step]
            elif len(preds) < num_missing:
                preds += [preds[-1]] * (num_missing - len(preds))

            # Fill in reconstructed frames
            for j, frame in enumerate(preds):
                full_frames[s_idx + j + 1] = frame

        print(f"Reconstructed {sum(f is not None for f in full_frames)}/{len(frames)} frames.")
        return full_frames

    def recursive_interpolate(self, f_start, f_end, depth):
        """Recursively interpolate frames between two inputs."""
        if depth == 0:
            return []

        mid_pred = self.interpolator.predict_between(f_start, f_end, 1)[0][1].numpy().astype(np.uint8)
        return (
            self.recursive_interpolate(f_start, mid_pred, depth - 1)
            + [mid_pred]
            + self.recursive_interpolate(mid_pred, f_end, depth - 1)
        )

    # ------------------------- #
    # Video Comparison
    # ------------------------- #
    def compare_videos(self, original_frames, reconstructed_frames, fps):
        """Creates side-by-side comparison video."""
        min_len = min(len(original_frames), len(reconstructed_frames))
        print(f"Comparing {min_len} frames (Original vs Reconstructed)")
        side_by_side = []

        for i in tqdm(range(min_len), desc="Rendering comparison", ncols=90):
            orig, recon = original_frames[i], reconstructed_frames[i]
            if recon is None:
                recon = np.zeros_like(orig)
            if orig.shape != recon.shape:
                recon = cv2.resize(recon, (orig.shape[1], orig.shape[0]))
            side_by_side.append(np.concatenate([orig, recon], axis=1))

        self.play_video(side_by_side, fps, title="comparison_video")

    # ------------------------- #
    # Metrics Computation
    # ------------------------- #
    def reconstruct_metrics(self, frames, reconstructed_frames, csv_path):
        print(len(frames), len(reconstructed_frames))
        
        """Compute MSE, invSSIM, LPIPS for reconstructed frames."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device} for LPIPS computation.")
        loss_fn = load_lpips_model(device)

        df = pd.read_csv(csv_path)
        available = set(df['Frame_Index'].tolist())
        metrics = []

        print("Computing frame-wise metrics ...")
        count = 0
        mean_mse = 0.0
        mean_inv_ssim = 0.0
        mean_lpips = 0.0
        mean_metric = 0.0

        for i, (orig, recon) in tqdm(enumerate(zip(frames, reconstructed_frames)), total=len(frames), ncols=90):
            if i in available or recon is None:
                continue
            if orig.shape != recon.shape:
                recon = cv2.resize(recon, (orig.shape[1], orig.shape[0]))

            mse = compute_mse(orig, recon)
            inv_ssim = compute_inv_ssim(orig, recon)
            lpips_val = compute_lpips(orig, recon, loss_fn, device)

            count += 1
            mean_mse += (mse - mean_mse) / count
            mean_inv_ssim += (inv_ssim - mean_inv_ssim) / count
            mean_lpips += (lpips_val - mean_lpips) / count
            mean_metric += ((0.5*mse + 0.3*inv_ssim + 0.2*lpips_val) - mean_metric) / count

            metrics.append((i, mean_mse, mean_inv_ssim, mean_lpips, mean_metric))

        print(f"Metrics computed for {len(metrics)} frames.")
        return (i, mean_mse, mean_inv_ssim, mean_lpips, mean_metric)