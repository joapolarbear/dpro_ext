import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights

# data
def cnn_dummpy_data(batch_size):
    # create dummy data for training
    x_train = np.ones((batch_size, 3, 224, 224), dtype=np.float32)
    y_train = np.ones((batch_size, 1), dtype=np.float32)
    return x_train, y_train