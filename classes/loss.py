import torchmetrics
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim_metric

class SSIM_MSE_Loss(nn.Module):
    def __init__(self, alpha=0.85):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_val = ssim_metric(pred, target, data_range=1.0)
        ssim_loss = 1 - ssim_val
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss