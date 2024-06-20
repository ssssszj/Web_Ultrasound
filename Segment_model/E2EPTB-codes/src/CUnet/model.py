import torch
import torch.nn as nn
import torchviz
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import models
import os


class ConvBlock(nn.Module):
    """
    Convolutional block.
    """
    def __init__(self, input_channels: int = None, output_channels: int = None, stride=1):
        """

        Args:
            input_channels:
            output_channels:
        """
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(num_features=output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(num_features=output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """

        Args:
            x:

        Returns:

        """
        x = self.conv(x)
        return x


class CUnet(nn.Module):
    """CU-Net model."""
    def __init__(self, input_channels: int = 3,
                 output_channels: int = 1,
                 number_of_class: int = 2,
                 features: int = 32):
        """

        Args:
            input_channels:
            output_channels:
            features:
        """
        super(CUnet, self).__init__()

        self.features = features

        # Encoding block
        self.encoder1 = ConvBlock(input_channels=input_channels, output_channels=self.features, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ConvBlock(input_channels=self.features, output_channels=self.features * 2, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ConvBlock(input_channels=self.features * 2, output_channels=self.features * 4, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ConvBlock(input_channels=self.features * 4, output_channels=self.features * 8, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bootleneck
        self.bootleneck = ConvBlock(input_channels=self.features * 8, output_channels=self.features * 16)

        # Decoding block
        self.upconv4 = nn.ConvTranspose2d(in_channels=self.features * 16, out_channels=self.features * 8, kernel_size=2,
                                          stride=2)
        self.decoder4 = ConvBlock(input_channels=self.features * 16, output_channels=self.features * 8)
        self.upconv3 = nn.ConvTranspose2d(in_channels=self.features * 8, out_channels=self.features * 4, kernel_size=2,
                                          stride=2)
        self.decoder3 = ConvBlock(input_channels=self.features * 8, output_channels=self.features * 4)
        self.upconv2 = nn.ConvTranspose2d(in_channels=self.features * 4, out_channels=self.features * 2, kernel_size=2,
                                          stride=2)
        self.decoder2 = ConvBlock(input_channels=self.features * 4, output_channels=self.features * 2)
        self.upconv1 = nn.ConvTranspose2d(in_channels=self.features * 2, out_channels=self.features, kernel_size=2,
                                          stride=2)
        self.decoder1 = ConvBlock(input_channels=self.features * 2, output_channels=self.features)

        self.conv = nn.Conv2d(in_channels=self.features, out_channels=output_channels, kernel_size=1, stride=1,
                              padding=0)

        # Classifier
        self.global_Avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.features * 16, number_of_class)

    def forward(self, x):
        """

        Args:
            x:

        Returns:

        """
        encoding1 = self.encoder1(x)
        pool1 = self.pool1(encoding1)
        encoding2 = self.encoder2(pool1)
        pool2 = self.pool2(encoding2)
        encoding3 = self.encoder3(pool2)
        pool3 = self.pool3(encoding3)
        encoding4 = self.encoder4(pool3)
        pool4 = self.pool4(encoding4)

        bottleneck = self.bootleneck(pool4)
        
        label = self.global_Avg(bottleneck)
        label = label.view(label.size(0), -1)
        label = self.fc(label)

        decoding4 = self.upconv4(bottleneck)
        decoding4 = torch.cat([decoding4, encoding4], dim=1)
        decoding4 = self.decoder4(decoding4)
        decoding3 = self.upconv3(decoding4)
        decoding3 = torch.cat([decoding3, encoding3], dim=1)
        decoding3 = self.decoder3(decoding3)
        decoding2 = self.upconv2(decoding3)
        decoding2 = torch.cat([decoding2, encoding2], dim=1)
        decoding2 = self.decoder2(decoding2)
        decoding1 = self.upconv1(decoding2)
        decoding1 = torch.cat([decoding1, encoding1], dim=1)
        decoding1 = self.decoder1(decoding1)

        return torch.sigmoid(self.conv(decoding1)), torch.sigmoid(label)

class GradCamCUnet(nn.Module):
    def __init__(self, model):
        super(GradCamCUnet, self).__init__()

        self.unet = model
        self.unet.eval()
        self.gradients = None
        self.activation = {}

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
    
    def register_hook(node, name):
        node.register_forward_hook(get_activation(name))

    def get_activations(self, x):
        encoding1 = self.unet.encoder1(x)
        pool1 = self.unet.pool1(encoding1)
        encoding2 = self.unet.encoder2(pool1)
        pool2 = self.unet.pool2(encoding2)
        encoding3 = self.unet.encoder3(pool2)
        pool3 = self.unet.pool3(encoding3)
        encoding4 = self.unet.encoder4(pool3)
        pool4 = self.unet.pool4(encoding4)

        return self.unet.bootleneck(pool4)

    def forward(self, x):
        encoding1 = self.unet.encoder1(x)
        pool1 = self.unet.pool1(encoding1)
        encoding2 = self.unet.encoder2(pool1)
        pool2 = self.unet.pool2(encoding2)
        encoding3 = self.unet.encoder3(pool2)
        pool3 = self.unet.pool3(encoding3)
        encoding4 = self.unet.encoder4(pool3)
        pool4 = self.unet.pool4(encoding4)

        bottleneck = self.unet.bootleneck(pool4)

        h = bottleneck.register_hook(self.activations_hook)
        
        label = self.unet.global_Avg(bottleneck)
        label = label.view(label.size(0), -1)
        label = self.unet.fc(label)

        # print(bottleneck.shape)
        decoding4 = self.unet.upconv4(bottleneck)
        # print(decoding4.shape)
        decoding4 = torch.cat([decoding4, encoding4], dim=1)
        # print(decoding4.shape)
        decoding4 = self.unet.decoder4(decoding4)
        decoding3 = self.unet.upconv3(decoding4)
        decoding3 = torch.cat([decoding3, encoding3], dim=1)
        decoding3 = self.unet.decoder3(decoding3)
        decoding2 = self.unet.upconv2(decoding3)
        decoding2 = torch.cat([decoding2, encoding2], dim=1)
        decoding2 = self.unet.decoder2(decoding2)
        decoding1 = self.unet.upconv1(decoding2)
        decoding1 = torch.cat([decoding1, encoding1], dim=1)
        decoding1 = self.unet.decoder1(decoding1)

        return torch.sigmoid(self.unet.conv(decoding1)), torch.sigmoid(label)


def visualize_model(model):
    # Create random input and output data
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    dg = torchviz.make_dot(y)
    file_name = model.model_name + "_model.gv"
    dg.render(os.path.join('../plots', file_name), view=False)

    return

if __name__ == "__main__":
    # unet = CUnet()
    # visualize_model(unet)

    inputs = torch.randn(1, 3, 224, 224)
    net = CUnet()
    net.eval()
    mask, label = net(Variable(inputs))
    print(mask, label)
    # torch.save(net.state_dict(), "model.pt")
    # g = torchviz.make_dot(y)
    # g.view()

    