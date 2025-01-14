import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import generate_conv_net

# A small size auxiliary model to determine whether the current frame has a vehicle to detect
class CNNModel(nn.Module):
    def __init__(self, input_shape,
                 nb_classes, nb_dense,
                 nb_filters, nb_layers):
        super().__init__()
        self.small_conv_net = generate_conv_net(input_shape, 
                                                nb_filters=nb_filters,
                                                nb_layers=nb_layers)
        flatten_size = self.small_conv_net(torch.zeros(1, *input_shape)).shape[1]
        self.predict = nn.Sequential(
            nn.Linear(flatten_size, nb_dense),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(nb_dense, nb_classes),
        )
    
    def forward(self, x):
        x = self.small_conv_net(x)
        x = self.predict(x)
        return x