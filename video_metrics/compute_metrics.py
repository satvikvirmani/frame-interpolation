import cv2
import csv
import os
import torch
from tqdm import tqdm
from metrics_utils import compute_mse, compute_inv_ssim, compute_lpips, load_lpips_model

def compute_video_metrics(video_path, output_path):
    """Compute MSE, SSIM, LPIPS between consecutive frames and save to CSV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO: Using device: {device} for LPIPS computation.")

    loss_fn_alex = load_lpips_model(device)
    print("INFO: LPIPS model loaded.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_pairs = total_frames - 1 if total_frames > 0 else None

    try:
        with open(output_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            header = ["Frame_Pair", "MSE", "Inverse SSIM", "LPIPS"]
            csv_writer.writerow(header)
            print(f"INFO: Output will be saved to '{output_path}'")

            ret, prev_frame = cap.read()
            if not ret:
                print("Error: Could not read the first frame.")
                cap.release()
                return

            frame_count = 1
            print("\n--- Starting Frame-by-Frame Metric Computation ---")

            with tqdm(total=total_pairs, desc="Processing Frame Pairs", ncols=100) as pbar:
                while True:
                    ret, curr_frame = cap.read()
                    if not ret:
                        break

                    mse_val = compute_mse(prev_frame, curr_frame)
                    inv_ssim_val = compute_inv_ssim(prev_frame, curr_frame)
                    lpips_val = compute_lpips(prev_frame, curr_frame, loss_fn_alex, device)

                    frame_pair_label = f"{frame_count}_vs_{frame_count + 1}"
                    csv_writer.writerow([
                        frame_pair_label,
                        f"{mse_val:.6f}",
                        f"{inv_ssim_val:.6f}",
                        f"{lpips_val:.6f}"
                    ])

                    prev_frame = curr_frame
                    frame_count += 1
                    pbar.update(1)

    except IOError as e:
        print(f"Error writing CSV: {e}")
    finally:
        cap.release()
        print(f"\n--- Video processing complete. Results saved to '{os.path.abspath(output_path)}'. ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute frame-to-frame video metrics.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("-o", "--output", type=str, default="metrics_output.csv",
                        help="Output CSV path (default: metrics_output.csv)")
    args = parser.parse_args()

    compute_video_metrics(args.video_path, args.output)