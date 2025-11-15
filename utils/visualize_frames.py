import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from metrics_utils import combined_difference


def visualize_frames_fullscreen(video_path, csv_path, start_frame=500, n=40, skip=0, cmap_name="coolwarm"):
    """
    Display frames in fullscreen grid with minimal whitespace and color-coded ΔΔ differences.
    """

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- Load metrics from CSV ---
    frame_diffs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mse = float(row["MSE"])
            inv_ssim = float(row["Inverse SSIM"])
            lpips = float(row["LPIPS"])
            diff = combined_difference(mse, inv_ssim, lpips)
            frame_diffs.append(diff)

    if len(frame_diffs) < 2:
        print("Error: Not enough frame data in CSV.")
        return

    # Normalize Δ for color mapping
    delta_min, delta_max = np.min(frame_diffs), np.max(frame_diffs)
    norm = colors.Normalize(vmin=delta_min, vmax=delta_max)
    cmap = cm.get_cmap(cmap_name)
    color_mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    # --- Frame selection ---
    if start_frame + n * (skip + 1) >= total_frames:
        n = (total_frames - start_frame - 1) // (skip + 1)
        print(f"Adjusted n to {n} due to video length.")

    frame_indices = [start_frame + i * (skip + 1) for i in range(n)]

    frames, deltas = [], []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        deltas.append(frame_diffs[idx])

    cap.release()

    # --- Optimal grid layout (close to square) ---
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # --- Create fullscreen figure ---
    plt.figure(figsize=(cols * 3.5, rows * 3.5))
    fig_manager = plt.get_current_fig_manager()
    try:
        fig_manager.full_screen_toggle()  # Works in most backends
    except Exception:
        pass

    # --- Plot frames ---
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(frames[i])
        ax.axis("off")

        delta = deltas[i]
        rgba = color_mapper.to_rgba(delta)

        # Add thin colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(rgba)
            spine.set_linewidth(6)

        # Add concise label
        ax.set_title(
            f"F{frame_indices[i]} Δ={delta:+.3e}",
            fontsize=8,
            color="white" if np.mean(rgba[:3]) < 0.4 else "black",
            backgroundcolor=rgba,
            pad=2
        )

    # --- Colorbar ---
    cbar = plt.colorbar(
        color_mapper, ax=plt.gcf().axes, orientation="horizontal", fraction=0.03, pad=0.02
    )
    cbar.set_label("Δ (Change in Difference) — Motion Intensity", fontsize=10)

    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05, wspace=0.02, hspace=0.05)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Fullscreen visualization of frames color-coded by ΔΔ motion intensity."
    )
    parser.add_argument("video_path", type=str, help="Path to input video.")
    parser.add_argument("csv_path", type=str, help="Path to metrics CSV.")
    parser.add_argument("-k", "--start", type=int, default=500, help="Starting frame index.")
    parser.add_argument("-n", "--num", type=int, default=40, help="Number of frames to display.")
    parser.add_argument("-s", "--skip", type=int, default=0, help="Skip every p frames between selections.")
    parser.add_argument("-c", "--cmap", type=str, default="coolwarm", help="Matplotlib colormap (default: coolwarm)")
    args = parser.parse_args()

    visualize_frames_fullscreen(
        args.video_path,
        args.csv_path,
        start_frame=args.start,
        n=args.num,
        skip=args.skip,
        cmap_name=args.cmap
    )