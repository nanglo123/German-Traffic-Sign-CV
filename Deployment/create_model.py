import torch
import torchvision
from model_class import CNNTraffic
from torch import nn
from torchvision import transforms

def create_CNN(seed:int=42):

    model = CNNTraffic(input_shape=3,output_shape=43)

    
    for param in model.parameters():
        param.requires_grad = False
    
    
    
    return model
