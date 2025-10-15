import os
import subprocess

steps = [
    ("Process Videos", "python process_videos.py"),
    ("Train Model", "python model_train.py"),
    ("Visualize Prediction", "python view_prediction.py")
]

if __name__ == "__main__":
    print("🚀 Starting Frame Interpolation Pipeline...\n")
    for name, cmd in steps:
        print(f"=== {name} ===")
        subprocess.run(cmd, shell=True)
        print("\n")
    print("✅ Pipeline complete.")