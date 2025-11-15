import os
import tensorflow as tf
from video_reconstrcutor import VideoReconstructor
import matplotlib.pyplot as plt
from model import build_improved_model
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from statsmodels.nonparametric.smoothers_lowess import lowess

# def plot_distribution(parameter, metric, retained_ratios):
#     plt.figure(figsize=(12,6))

#     # --- Line + points ---
#     plt.plot(metric, retained_ratios, '-', color='blue', lw=2, label="% frames retained")
#     plt.scatter(metric, retained_ratios, s=15, color='blue', alpha=0.7)

#     # --- Bell curve for metric ---
#     mu, sigma = np.mean(metric), np.std(metric)
#     x_vals = np.linspace(min(metric), max(metric), 400)
#     y_vals = norm.pdf(x_vals, mu, sigma)  # normal PDF
#     y_scaled = y_vals / max(y_vals) * max(retained_ratios)  # scale to match y range
#     plt.fill_between(x_vals, y_scaled, alpha=0.12, color='orange', label="Metric Distribution (scaled)")

#     # --- Aesthetics ---
#     plt.title(f"Frames Retained vs {parameter} Metric", fontsize=14, weight='bold')
#     plt.xlabel(f"{parameter} Metric", fontsize=12)
#     plt.ylabel("Frames Retained (%)", fontsize=12)
#     plt.grid(alpha=0.3, linestyle='--')

#     # More ticks
#     plt.xticks(np.linspace(min(metric), max(metric), 12))
#     plt.yticks(np.linspace(0, 100, 11))

#     plt.legend(frameon=False)
#     plt.tight_layout()
#     plot_path = os.path.join("plots", f"{parameter.lower()}_metric_distribution.png")
#     plt.savefig(plot_path, dpi=300)

def plot_distribution(parameter, metric, retained):
    plt.figure(figsize=(12,6))

    # --- Scatter ---
    plt.scatter(metric, retained, s=25, alpha=0.5, label="Data")

    # --- LOWESS smooth curve ---
    smoothed = lowess(retained, metric, frac=0.3)
    plt.plot(smoothed[:,0], smoothed[:,1], lw=3, label="LOWESS Trend")

    # --- Metric histogram (helps show distribution) ---
    plt.twinx()
    plt.hist(metric, bins=20, alpha=0.15, color='orange')
    plt.ylabel("Metric Frequency")

    # --- Aesthetics ---
    plt.title(f"{parameter} vs Frame Retention (%)", fontsize=15, weight='bold')
    plt.xlabel(f"{parameter} Metric", fontsize=12)
    plt.ylabel("Frames Retained (%)", fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"{parameter.lower()}_scatter.png"), dpi=300)
    
    plt.figure(figsize=(12,6))
    plt.hexbin(metric, retained, gridsize=30, cmap='Blues')
    plt.colorbar(label="Density")
    plt.xlabel(f"{parameter} Metric")
    plt.ylabel("Frames Retained (%)")
    plt.title(f"{parameter} vs Frame Retention — Density Map")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"{parameter.lower()}_density.png"), dpi=300)

INITIAL_LEARNING_RATE = 2e-4
CHECKPOINT_DIR="./checkpoints"
VIDEO_PATH = "../video/videos/videoplayback2.mp4"
KEYFRAMES_PATH = "../utils/keyframes/"

model = build_improved_model()

lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=INITIAL_LEARNING_RATE,
    first_decay_steps=282 * 5,  # Restart every 5 epochs
    t_mul=2.0,
    m_mul=0.9,
    alpha=0.1
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f"Restored from {ckpt_manager.latest_checkpoint}")
else:
    print("No checkpoint found.")

reconstructor = VideoReconstructor(model)

frames, fps = reconstructor.read_video(VIDEO_PATH)
print(f"Video has {len(frames)} frames at {fps:.2f} FPS.")

os.makedirs("plots", exist_ok=True)

mean_mse_s = []
mean_inv_ssim_s = []
mean_lpips_s = []
mean_pct_s = []
mean_metric_s = []

for filename in tqdm(os.listdir(KEYFRAMES_PATH)):
    csv_path = os.path.join(KEYFRAMES_PATH, filename)
    if os.path.isfile(csv_path):
        reconstructed_frames = reconstructor.reconstruct_frames(frames, csv_path)
        new_fps = fps * (len(reconstructed_frames) / len(frames))
        i, mean_mse, mean_inv_ssim, mean_lpips, mean_metric = reconstructor.reconstruct_metrics(frames, reconstructed_frames, csv_path)
        
        mean_mse_s.append(mean_mse)
        mean_inv_ssim_s.append(mean_inv_ssim)
        mean_lpips_s.append(mean_lpips)
        mean_metric_s.append(mean_metric)
        
        name_only = filename.replace(".csv", "")
        
        pct = float(name_only.split("_")[-1])
        mean_pct_s.append(pct)
        
        valid_count = sum(f is not None for f in reconstructed_frames)
        print(f"Reconstructed {valid_count}/{len(reconstructed_frames)} frames.")
        
plot_distribution("MSE", mean_mse_s, mean_pct_s)
plot_distribution("INV SSIM", mean_inv_ssim_s, mean_pct_s)
plot_distribution("LPIPS", mean_lpips_s, mean_pct_s)
plot_distribution("Combined Metric", mean_metric_s, mean_pct_s)