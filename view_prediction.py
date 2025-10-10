# view_prediction.py
import os
import torch
import cv2
from torchvision import transforms
from PIL import Image
from model_train import FrameInterpolationModel

def predict_between(frame1_path, frame2_path, model_path="models/frame_interpolation.pth"):
    model = FrameInterpolationModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    transform = transforms.ToTensor()
    frame1, frame2 = map(lambda p: transform(Image.open(p)).unsqueeze(0), [frame1_path, frame2_path])
    with torch.no_grad():
        pred = model(torch.cat((frame1, frame2), 1))
    return pred.squeeze(0).permute(1,2,0).numpy()

def show_result(f1, pred, f2):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    for i, img in enumerate([f1, pred, f2]):
        plt.subplot(1,3,i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.suptitle("Before → Predicted → After")
    plt.show()

if __name__ == "__main__":
    video_dir = "data/frames/Free Footage Sunrise Timelapse_720p"
    frames = sorted(os.listdir(video_dir))
    for idx in [5,10,15,20,25,30]:
        if frames[idx] != f"frame{idx+1:04d}.jpg":
            raise ValueError("Frame naming mismatch. Check the frame files.")
        f1, f2 = os.path.join(video_dir, frames[idx]), os.path.join(video_dir, frames[idx+2])
        pred_frame = predict_between(f1, f2)
        show_result(cv2.imread(f1), (pred_frame*255).astype("uint8"), cv2.imread(f2))