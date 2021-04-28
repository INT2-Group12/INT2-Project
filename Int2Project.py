import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

dataset = CIFAR10(root='data/', download=True, train=True, transform=ToTensor())
test_dataset = CIFAR10(root='data/', download=True, train=False, transform=ToTensor())
