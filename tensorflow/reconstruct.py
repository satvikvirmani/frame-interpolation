import os
import tensorflow as tf
from video_reconstrcutor import VideoReconstructor
from model import build_improved_model

INITIAL_LEARNING_RATE = 2e-4
CHECKPOINT_DIR="./checkpoints"
VIDEO_PATH = "../video/videos/videoplayback2.mp4"
CSV_PATH = "../utils/keyframes/keyframe_23.0625_1.7829_42.77.csv"

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

reconstructed_frames = reconstructor.reconstruct_frames(frames, CSV_PATH)
new_fps = fps * (len(reconstructed_frames) / len(frames))
reconstructor.play_video(reconstructed_frames, new_fps, title="Smooth Reconstructed Video")
reconstructor.compare_videos(frames, reconstructed_frames, fps)
metrics = reconstructor.reconstruct_metrics(frames, reconstructed_frames, CSV_PATH)

valid_count = sum(f is not None for f in reconstructed_frames)
print(f"Reconstructed {valid_count}/{len(reconstructed_frames)} frames.")