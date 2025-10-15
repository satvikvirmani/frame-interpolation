import torch
import torch.nn as nn

class FrameInterpolationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(6, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(inplace=True)
            ) for _ in range(3)
        ])
        self.resize = nn.Upsample(size=(128, 160), mode='bilinear', align_corners=True)
        self.fusion_conv = nn.Conv2d(384, 128, 3, 1, 1)
        self.upsample_conv = nn.ConvTranspose2d(128, 3, 3, 2, 1, 1, 1)

    def forward(self, x):
        features = [f(x) for f in self.feature_extractor]
        resized = [self.resize(f) for f in features]
        x = torch.cat(resized, dim=1)
        x = torch.relu(self.fusion_conv(x))
        return self.upsample_conv(x)