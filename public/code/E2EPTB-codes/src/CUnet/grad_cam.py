
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import GradCamCUnet
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import make_grid
from data_loader import CervixDataset
from utils import get_samples_for_label, show_images
    
def get_gradcam(model, image, label): # 1 - PRETERM, 0 - CONTROL
    orginal_shape = image.size
    img = image.resize((256, 256))
    inputs = Variable(transforms.ToTensor()(img).unsqueeze(0))
    pred_mask, labels = model(inputs)
    
    # return labels[0, 0] > labels[0, 1]

    print(labels)
    labels[:, label].backward()

    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations(inputs).detach()
    
    for i in range(len(pooled_gradients)):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=1).squeeze()

    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.cpu().numpy()
    # plt.imshow(heatmap, interpolation='nearest')
    # plt.show()
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    heatmap = cv2.resize(heatmap, (orginal_shape[0], orginal_shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    outputs = pred_mask.round().squeeze(0).cpu().data
    mask = transforms.ToPILImage()(outputs)
    mask = cv2.cvtColor(np.array(mask), cv2.COLOR_GRAY2RGB)
    mask = cv2.resize(mask, (orginal_shape[0], orginal_shape[1]))

    # img = cv2.addWeighted(mask, 0.2, img, 0.8, 0)
    superimposed_img = cv2.addWeighted(heatmap, 0.3, img, 0.7, 0)
    superimposed_img = cv2.hconcat([img, superimposed_img])

    cv2.imshow('image', superimposed_img)
    cv2.waitKey(0)
    
def vizualize_features(model, image, names):
    # https://discuss.pytorch.org/t/visualize-feature-map/29597/4
        
    orginal_shape = image.size
    img = image.resize((256, 256))
    inputs = Variable(transforms.ToTensor()(img).unsqueeze(0))
    _, _ = model(inputs)
    
    for name in names:
        act = model.activation[name].squeeze()
        show_images(act, name)
    plt.show()


if __name__ == "__main__":
    unet = torch.load('../models/unet_.pt', map_location='cpu')
    # filters = unet.encoder1.conv[3].weight.data
    # fig = plt.figure()
    # plt.figure(figsize=(10,10))
    # fig, axarr = plt.subplots(filters.size(0))
    # for idx in range(filters.size(0)):
    #     plt.subplot(4,8, idx + 1)
    #     plt.imshow(filters[idx, 1, :])
    #     plt.axis('off')
    # fig.show()
    
    # kernels = unet.bootleneck.conv[3].weight.detach().clone()
    # kernels = kernels - kernels.min()
    # kernels = kernels / kernels.max()
    # print(kernels.shape)
    # img = make_grid(kernels)
    # plt.imshow(img.permute(1, 2, 0))
    
    layers = [unet.encoder1, unet.encoder2, unet.encoder3, unet.encoder4, unet.bootleneck]
    # for layer in layers:
    #     print(layer.conv[0].weight[:,0,:,:].unsqueeze(dim=1).shape)
    #     print(layer.conv[0].weight.data.shape)
    #     print(layer.conv[3].weight.data.shape)
    # weight = unet.encoder1.conv[3].weight.data.numpy()
    # plt.imshow(weight[0, ...])
    # plt.show()
    
    cervix_dataset = CervixDataset("../data/Bezier", "../data/Beziermask", "../data/annotations_final.csv")
    gradCamUnet = GradCamCUnet(unet)
    gradCamUnet.unet.encoder1.conv[0].register_forward_hook(gradCamUnet.get_activation('en1_conv0'))
    gradCamUnet.unet.encoder1.conv[3].register_forward_hook(gradCamUnet.get_activation('en1_conv1'))
    gradCamUnet.unet.encoder2.conv[0].register_forward_hook(gradCamUnet.get_activation('en2_conv0'))
    gradCamUnet.unet.encoder2.conv[3].register_forward_hook(gradCamUnet.get_activation('en2_conv1'))
    gradCamUnet.unet.encoder3.conv[0].register_forward_hook(gradCamUnet.get_activation('en3_conv0'))
    gradCamUnet.unet.encoder3.conv[3].register_forward_hook(gradCamUnet.get_activation('en3_conv1'))
    gradCamUnet.unet.encoder4.conv[0].register_forward_hook(gradCamUnet.get_activation('en4_conv0'))
    gradCamUnet.unet.encoder4.conv[3].register_forward_hook(gradCamUnet.get_activation('en4_conv1'))
    gradCamUnet.unet.bootleneck.conv[0].register_forward_hook(gradCamUnet.get_activation('bootleneck_conv0'))
    gradCamUnet.unet.bootleneck.conv[3].register_forward_hook(gradCamUnet.get_activation('bootleneck_conv1'))
    
    images = get_samples_for_label(cervix_dataset, label=1)
    images = [images[i] for i in [30, 31, 34, 35]] # PRETERM
    # images = [images[i] for i in [5,6,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]] # CONTROL
    for i, image in enumerate(images):
        vizualize_features(gradCamUnet, image, ['en1_conv0', 'en1_conv1', 'en2_conv0', 'en2_conv1', 
                                                'en3_conv0', 'en3_conv1', 'en4_conv0', 'en4_conv1',
                                                'bootleneck_conv0', 'bootleneck_conv1'])
        get_gradcam(gradCamUnet, image, label=1)
            # print(i)