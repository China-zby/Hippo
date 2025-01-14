import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_5layers = nn.Sequential(
            Conv2dSame(3, 32, 4, 2),
            nn.ReLU(),
            Conv2dSame(32, 64, 4, 2),
            nn.ReLU(),
            Conv2dSame(64, 64, 4, 2),
            nn.ReLU(),
            Conv2dSame(64, 64, 4, 2),
            nn.ReLU(),
            Conv2dSame(64, 64, 4, 2),
            nn.ReLU()
        )
        self.decode_layer = Conv2dSame(64, 64, 4)
        self.pre_conv = Conv2dSame(64, 1, 4)
        
    def forward(self, x):
        x = self.conv_5layers(x)
        x = self.decode_layer(x)
        x = F.relu(x)
        x = self.pre_conv(x)[:, 0, :, :]
        x = torch.sigmoid(x)
        return x