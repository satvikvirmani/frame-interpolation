import csv
import numpy as np
import argparse
import os

def compute_differences(csv_path):
    """Read metrics from CSV and return arrays."""
    mse, inv_ssim, lpips = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mse.append(float(row["MSE"]))
            inv_ssim.append(float(row["Inverse SSIM"]))
            lpips.append(float(row["LPIPS"]))
    return np.array(mse), np.array(inv_ssim), np.array(lpips)

def combined_difference(mse, inv_ssim, lpips):
    """Weighted combination metric."""
    return 0.5 * mse + 0.3 * inv_ssim + 0.2 * lpips

def select_keyframes(csv_path, abs_thresh=None, delta_thresh=None, adapt_factor=1.0, save_path=None, verbose=True):
    """Select keyframes based on difference thresholds and always include first + last frame."""
    
    mse, inv_ssim, lpips = compute_differences(csv_path)
    diffs = combined_difference(mse, inv_ssim, lpips)
    delta_diffs = np.diff(diffs, prepend=diffs[0])

    # --- Auto thresholds if not provided ---
    if abs_thresh is None:
        abs_thresh = np.mean(diffs) + adapt_factor * np.std(diffs)
    if delta_thresh is None:
        delta_thresh = np.mean(np.abs(delta_diffs)) + adapt_factor * np.std(np.abs(delta_diffs))

    # --- Selection logic ---
    keep_indices = [
        i for i in range(len(diffs))
        if (diffs[i] > abs_thresh) and (abs(delta_diffs[i]) > delta_thresh)
    ]

    # --- Always include first and last frames ---
    if 0 not in keep_indices:
        keep_indices.insert(0, 0)
    if (len(diffs) - 1) not in keep_indices:
        keep_indices.append(len(diffs) - 1)
    keep_indices = sorted(set(keep_indices))

    if(verbose):
        print(f"Absolute: {abs_thresh:.2f}, Delta: {delta_thresh:.2f} -> Retained: {len(keep_indices)}/{len(diffs)} frames ({len(keep_indices)/len(diffs)*100:.2f})")
    
    # --- Optionally save selected indices ---
    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".csv":
            with open(save_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Frame_Index"])
                for idx in keep_indices:
                    writer.writerow([idx])
        else:
            np.savetxt(save_path, keep_indices, fmt="%d")
        print(f"💾 Saved selected frame indices to {os.path.abspath(save_path)}")

    return keep_indices, len(keep_indices)/len(diffs)*100, (np.mean(diffs), np.std(diffs)), (np.mean(np.abs(delta_diffs)), np.std(np.abs(delta_diffs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select key frames for efficient frame interpolation or transfer."
    )
    parser.add_argument("csv_path", type=str, help="Path to metrics CSV file.")
    parser.add_argument("-a", "--abs_thresh", type=float, default=None, help="Absolute difference threshold.")
    parser.add_argument("-d", "--delta_thresh", type=float, default=None, help="Delta difference threshold.")
    parser.add_argument("--adapt", type=float, default=1.0, help="Adaptive scaling factor (default 1.0).")
    parser.add_argument("-o", "--output", type=str, default=None, help="Optional path to save selected frame indices (txt/csv).")

    args = parser.parse_args()

    selected_frames = select_keyframes(
        args.csv_path,
        abs_thresh=args.abs_thresh,
        delta_thresh=args.delta_thresh,
        adapt_factor=args.adapt,
        save_path=args.output
    )

    print("\nSelected keyframe indices:")
    print(selected_frames)