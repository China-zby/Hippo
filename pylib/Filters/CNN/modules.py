import torch
from torch import nn
from torch.optim import RMSprop

def get_loss(regression):
    if regression:
        return nn.MSELoss()
    else:
        return nn.CrossEntropyLoss()

def get_optimizer(regression, parameters, nb_layers, lr_mult=1):
    if regression:
        return RMSprop(parameters, lr=0.001 / (1.5 * nb_layers) * lr_mult)
    else:
        return RMSprop(parameters, lr=0.001 * lr_mult) # / (5 * nb_layers))

class ConvNet(nn.Module):
    def __init__(self, 
                 input_shape,
                 nb_filters=32, 
                 nb_layers=1):
        super(ConvNet, self).__init__()

        assert nb_layers >= 0
        assert nb_layers <= 3
        
        layers = []
        layers.append(nn.Conv2d(input_shape[0], nb_filters, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(nb_filters, nb_filters, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Dropout(0.25))

        if nb_layers > 1:
            layers.append(nn.Conv2d(nb_filters, nb_filters * 2, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(nb_filters * 2, nb_filters * 2, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(0.25))

        if nb_layers > 2:
            layers.append(nn.Conv2d(nb_filters * 2, nb_filters * 4, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(nb_filters * 4, nb_filters * 4, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(0.25))

        layers.append(nn.Flatten())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def generate_conv_net(input_shape, 
                      nb_filters=32,
                      nb_layers=1):
    return ConvNet(input_shape,
                   nb_filters=nb_filters,
                   nb_layers=nb_layers)