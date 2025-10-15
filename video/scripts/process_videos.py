# process_videos.py
import os
import cv2

def process_video(input_path, output_path, max_frames=200, target_resolution=(160, 90)):
    cap = cv2.VideoCapture(input_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if width == 1280 and height == 720:
        os.makedirs(output_path, exist_ok=True)
        count, success = 0, True
        while success and count < max_frames:
            success, frame = cap.read()
            if not success: break
            count += 1
            resized = cv2.resize(frame, target_resolution)
            cv2.imwrite(os.path.join(output_path, f"frame{count:04d}.jpg"), resized)
        cap.release()
        print(f"✅ Processed {input_path}")
    else:
        print(f"⚠️ Skipped {input_path} (not 720p)")

if __name__ == "__main__":
    input_dir = "data/videos"
    output_dir = "data/frames"
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.endswith(".mp4"):
            video_name = os.path.splitext(fname)[0]
            process_video(os.path.join(input_dir, fname), os.path.join(output_dir, video_name))

    print("🎬 Video processing complete.")