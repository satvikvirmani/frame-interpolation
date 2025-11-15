import os
import subprocess
import sys
from pathlib import Path

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.mpeg', '.mpg', '.m4v'}
VIDEO_PATH = "video/videos"

def find_videos(root_dir, recursive=True):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    if recursive:
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                yield p
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                yield p


def probe_with_opencv(path, max_frames=500):
    try:
        import cv2
    except Exception:
        return None  # Indicate cv2 not available

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return set()  # empty set indicates failure to open

    dims = set()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        dims.add((w, h))
        frame_count += 1
        if frame_count >= max_frames:
            # stop early for long videos; assume sampled frames are representative
            break
    cap.release()
    return dims


def probe_with_ffprobe(path):
    # Use ffprobe to get stream width/height if possible
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(path)
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if len(lines) >= 2:
            w = int(lines[0])
            h = int(lines[1])
            return {(w, h)}
    except Exception:
        return set()
    return set()


def analyze_video(path):
    # Try OpenCV first
    cv_dims = probe_with_opencv(path)
    if cv_dims is None:
        # cv2 not installed -> fallback to ffprobe
        ff_dims = probe_with_ffprobe(path)
        return ff_dims, 'ffprobe'
    if len(cv_dims) == 0:
        # cv2 failed to open or returned empty -> try ffprobe
        ff_dims = probe_with_ffprobe(path)
        return (ff_dims or cv_dims), 'ffprobe' if ff_dims else 'opencv-failed'
    return cv_dims, 'opencv'

if __name__ == '__main__':
    if not os.path.exists(VIDEO_PATH):
        print(f"Videos directory does not exist.")
        sys.exit(1)

    results = []
    for video_path in find_videos(VIDEO_PATH):
        print(f"Processing: {video_path}")
        dims, method = analyze_video(video_path)
        if not dims:
            print(f"  -> No dimensions detected (method={method}). File may be unreadable or requires ffprobe/ffmpeg.")
        else:
            dims_list = sorted(dims)
            print(f"  -> Method: {method}; Unique dimensions ({len(dims_list)}): {dims_list}")
        results.append((str(video_path), method, sorted(list(dims))))