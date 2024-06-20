import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def jaccard(outputs, targets):
    outputs = outputs.view(outputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (outputs * targets).sum(1)
    union = (outputs + targets).sum(1) - intersection
    jac = (intersection + 0.001) / (union + 0.001)
    return jac.mean()

def metrics(y_true, y_pred, average=None):
    """
    
    Args:
        y_true: 
        y_pred: 
​
    Returns:
​
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)

    if average is None:
        precision = precision.mean()
        recall = recall.mean()
        f1 = f1.mean()
    
    return accuracy, precision, recall, f1

def conf_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    print(matrix)

def auc(y_true, y_pred, average=None):
    return roc_auc_score(y_true, y_pred, average=average)

def one_hot_vector(label, num_classes):
    y = torch.eye(num_classes)

    return y[label]

def splitDataset(items, v_size=0.2, t_size=0.2):
    trainSize = 1 - v_size - t_size
    assert(trainSize > 0)

    train_indices, val_indices = train_test_split(items, test_size=v_size, random_state=42)
    test_size = t_size / (1 - v_size)
    train_inds, test_inds = train_test_split(np.arange(len(train_indices)), test_size=test_size, random_state=42)
    test_indices = train_indices[test_inds]
    train_indices = train_indices[train_inds]

    print(len(train_indices), trainSize)
    print(len(val_indices), v_size)
    print(len(test_indices), t_size)

    return train_indices, val_indices, test_indices

def get_samples_for_label(dataset, num_of_samples=100, label=1): # 1 - PRETERM, 0 - CONTROL
    images = list()
    for i, (input, _, l) in enumerate(dataset):
        if l == label:
            images.append(input)
            if len(images) == num_of_samples:
                return images

    return images 

def get_samples_and_mask_for_label(dataset, num_of_samples=100, label=1): # 1 - PRETERM, 0 - CONTROL
    images = list()
    for i, (input, mask, l) in enumerate(dataset):
        if l == label:
            images.append(tuple((input, mask)))
            if len(images) == num_of_samples:
                return images

    return images

def show_images(images, title):
    n_images = len(images)
    size = images[0].shape[0]
    images_per_row = 20
    n_cols = n_images // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = images[col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image.numpy(), 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    # plt.title(title)
    plt.grid(False)
    plt.axis('off')
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig(title + ".jpg")
    
    # plt.show()