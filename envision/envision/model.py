import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyConvNet(nn.Module):
    """
    2-layer CNN: 1->8->16 channels, 3x3 kernels, no bias in conv layers.
    Designed to be small enough for full SVD analysis of every weight matrix.
    """

    def __init__(self):
        super().__init__()
        # bias=False so W alone captures the transformation (no affine shift)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=0, bias=False)   # (B,8,26,26)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=0, bias=False)  # (B,16,11,11)
        self.fc = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # (B, 8, 26, 26)
        x = F.max_pool2d(x, 2)          # (B, 8, 13, 13)
        x = F.relu(self.conv2(x))       # (B, 16, 11, 11)
        x = F.max_pool2d(x, 2)          # (B, 16,  5,  5)
        x = x.view(x.size(0), -1)       # (B, 400)
        return self.fc(x)               # (B, 10)

    def get_activations(self, x):
        """Return every intermediate tensor for analysis."""
        a1 = self.conv1(x)
        h1 = F.relu(a1)
        p1 = F.max_pool2d(h1, 2)
        a2 = self.conv2(p1)
        h2 = F.relu(a2)
        p2 = F.max_pool2d(h2, 2)
        flat = p2.view(p2.size(0), -1)
        out = self.fc(flat)
        return {
            'pre_relu1':  a1,
            'post_relu1': h1,
            'pool1':      p1,
            'pre_relu2':  a2,
            'post_relu2': h2,
            'pool2':      p2,
            'output':     out,
        }

    def weight_matrices(self):
        """Return (W1, W2) as 2-D numpy arrays: (out_ch, in_ch*kH*kW)."""
        import numpy as np
        W1 = self.conv1.weight.data.numpy()
        W2 = self.conv2.weight.data.numpy()
        W1 = W1.reshape(W1.shape[0], -1)   # (8,  9)
        W2 = W2.reshape(W2.shape[0], -1)   # (16, 72)
        return W1, W2
