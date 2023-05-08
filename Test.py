import torch
import torch.nn as nn
import torchvision
import torchvision.models.detection as detection
import torchvision.transforms as transforms
import csv
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import ToPILImage

import random
import sys
import matplotlib.pyplot as plt
from PIL import Image

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
# For training
images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 1, 4)
boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
labels = torch.randint(1, 91, (4, 1))
images = list(image for image in images)
targets = []
for i in range(len(images)):
    d = {}
    d['boxes'] = boxes[i]
    d['labels'] = labels[i]
    targets.append(d)
print(images)
print(targets)
output = model(images, targets)
print(output)
# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)
