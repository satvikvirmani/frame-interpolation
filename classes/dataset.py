import os
from PIL import Image
from torch.utils.data import Dataset

class FrameInterpolationDataset(Dataset):
    def __init__(self, root_dir, list_file, transform=None):
        """
        Args:
            root_dir (str): Root directory of the Vimeo-90K dataset (e.g. 'vimeo_triplet')
            list_file (str): Path to the txt file listing sequences (e.g. 'tri_trainlist.txt' or 'tri_testlist.txt')
            transform (callable, optional): Transform to apply on frames
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Read all folder names from the list file
        list_path = os.path.join(root_dir, list_file)
        with open(list_path, 'r') as f:
            self.samples = [line.strip() for line in f.readlines() if line.strip()]
        
        # Each sample points to something like '00001/0001'
        # and contains three frames: im1.png, im2.png, im3.png

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_path = os.path.join(self.root_dir, 'sequences', self.samples[idx])
        
        # Load three frames
        frame1_path = os.path.join(seq_path, 'im1.png')
        frame2_path = os.path.join(seq_path, 'im2.png')  # middle frame (target)
        frame3_path = os.path.join(seq_path, 'im3.png')
        
        # Open as PIL Images
        frame1 = Image.open(frame1_path).convert('RGB')
        frame2 = Image.open(frame2_path).convert('RGB')
        frame3 = Image.open(frame3_path).convert('RGB')

        # Apply transforms if provided
        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
            frame3 = self.transform(frame3)

        # Use im1, im3 as inputs and im2 as target
        return frame1, frame3, frame2
