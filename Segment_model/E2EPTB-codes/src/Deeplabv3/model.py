""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def createDeepLabv3(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=True, progress=True)
    # Added a Sigmoid activation after the last convolution layer
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model
