import numpy as np  # linear algebra
import torch
import torchvision.models.detection as detection
import torchvision.transforms as transforms
import csv
from torch.utils.data import Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import random
from engine import train_one_epoch
import matplotlib.pyplot as plt
from PIL import Image

import os
import re

resize = 600


# Load the data from asl-alphabet
def Dataset_Loader():
    x_data, x_label, y_data, y_label = [], [], [], []
    train_dataset = csv.reader(open('./ASL_Letters/train/_annotations.csv'))
    for line in train_dataset:
        if (line[0]) == 'filename':
            print('ASL_Letters/train/' + line[0])
            continue
        x_data.append('./ASL_Letters/train/' + line[0])
        bbox = []
        for i in range(4, 8):
            bbox.append(float(line[i]) * resize / float(line[1]))
        bbox.append(ord(line[3]) - 65)
        d = torch.tensor(bbox)
        x_label.append(d)
    test_dataset = csv.reader(open('./ASL_Letters/test/_annotations.csv'))
    for line in test_dataset:
        if (line[0]) == 'filename':
            print('ASL_Letters/test/' + line[0])
            continue
        y_data.append('./ASL_Letters/test/' + line[0])
        bbox = []
        for i in range(4, 8):
            bbox.append(float(line[i]) * resize / float(line[1]))
        bbox.append(ord(line[3]) - 65)
        d = torch.tensor(bbox)
        y_label.append(d)
    return x_data, x_label, y_data, y_label


train_data, train_label, test_data, test_label = Dataset_Loader()
print("Test data size: ", len(test_data), " Train data size: ", len(train_data))

batch_size = 4
learning_rate = 1e-6
decay = 1e-7

preprocess = transforms.Compose([
    transforms.Resize(resize),
    transforms.ToTensor()
])


def show_image(img, target, bbox, label, filename):
    toPIL = transforms.ToPILImage()
    tbox = target["boxes"].detach().numpy()[0]
    tlabel = target["labels"].detach().numpy()[0]
    trec = plt.Rectangle(
        xy=(tbox[0], tbox[1]),
        width=tbox[2] - tbox[0],
        height=tbox[3] - tbox[1],
        fill=False,
        edgecolor='blue',
        linewidth=2
    )
    rec = plt.Rectangle(
        xy=(bbox[0], bbox[1]),
        width=bbox[2] - bbox[0],
        height=bbox[3] - bbox[1],
        fill=False,
        edgecolor='red',
        linewidth=2
    )
    image = toPIL(img)
    fig = plt.imshow(image)
    fig.axes.add_patch(rec)
    fig.axes.add_patch(trec)
    text_color = 'w'
    fig.axes.text(rec.xy[0], rec.xy[1], label,
                  va='center', ha='center', fontsize=9, color=text_color,
                  bbox=dict(facecolor='red', lw=0))
    fig.axes.text(trec.xy[0] + trec.width, trec.xy[1] + trec.height, tlabel,
                  va='center', ha='center', fontsize=9, color=text_color,
                  bbox=dict(facecolor='blue', lw=0))
    plt.savefig('./Output/' + filename + ".png")
    plt.close()
    # plt.show()


def evaluate_IOU(target, bbox):
    tbox = target["boxes"].detach().numpy()[0]
    xy_1 = [max(bbox[0], tbox[0]), max(bbox[1], tbox[1])]
    xy_2 = [min(bbox[2], tbox[2]), min(bbox[3], tbox[3])]
    I = (xy_2[0] - xy_1[0]) * (xy_2[1] - xy_1[1])
    U = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) + (tbox[2] - tbox[0]) * (tbox[3] - tbox[1]) - I
    return I / U


def bbox_to_image(img, bbox, label, filename):
    toPIL = transforms.ToPILImage()
    rec = plt.Rectangle(
        xy=(bbox[0], bbox[1]),
        width=bbox[2] - bbox[0],
        height=bbox[3] - bbox[1],
        fill=False,
        edgecolor='red',
        linewidth=2
    )
    image = toPIL(img)
    fig = plt.imshow(image)
    fig.axes.add_patch(rec)
    text_color = 'w'
    fig.axes.text(rec.xy[0], rec.xy[1], label,
                  va='center', ha='center', fontsize=9, color=text_color,
                  bbox=dict(facecolor='red', lw=0))
    plt.savefig('./Output/' + filename)
    plt.close()
    # plt.show()


def test(dataset_loader, model, dev, e, ResNet):
    avg_IOU = 0
    avg_accuracy = 0
    avg_score = 0
    model.eval()
    print("--Test begins--")

    with torch.no_grad():
        image_num = list(random.sample(range(72), 5))
        if ResNet:
            path = './Output/ResNet_Epoch' + str(e)
        else:
            path = './Output/Mobilev3_Epoch' + str(e)
        os.mkdir(path)
        k = 0
        for i, data in enumerate(dataset_loader):
            images, targets = data
            images = images.to(dev)
            images = list(image for image in images)
            test_output = model(images)[0]
            boxes = test_output['boxes'].cpu().detach().numpy()
            labels = test_output['labels'].cpu().detach().numpy()
            scores = test_output['scores'].cpu().detach().numpy()
            obj_index = np.argwhere(scores > 0.8).squeeze(axis=1).tolist()
            if i in image_num and len(boxes) > 0:
                if ResNet:
                    show_image(images[0], targets, boxes[0], labels[0], "/ResNet_Epoch" + str(e) + "/Image" + str(i))
                else:
                    show_image(images[0], targets, boxes[0], labels[0], "/Mobilev3_Epoch" + str(e) + "/Image" + str(i))
            if len(boxes) > 0:
                k += 1
                avg_IOU += evaluate_IOU(targets, boxes[0])
                avg_accuracy += int(labels[0] == targets["labels"].detach().numpy()[0])
                avg_score += scores[0]
        print("avg_IOU: ", avg_IOU / k, " - avg_accuracy: ", avg_accuracy / 72, " - avg_score: ", avg_score / k)
        # return b_list, l_list, s_list, T_label


class ASLDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.target = labels

    def __getitem__(self, index):
        img_pil = Image.open(self.images[index])

        img_tensor = preprocess(img_pil)
        label = self.target[index]
        boxes = label[0:4]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = label[4]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([index], dtype=torch.int64)
        area = (boxes[3] - boxes[1]) * (boxes[2] - boxes[0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}
        return img_tensor, target

    def __len__(self):
        return len(self.images)

ResNet_available = True
if ResNet_available:
    weight_path = 'Faster_RCNN_ResNet.tar'
else:
    weight_path = 'Faster_RCNN_Mobile3.tar'

train_set = ASLDataset(train_data, train_label)
test_set = ASLDataset(test_data, test_label)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
test_image = test_loader.dataset[0]
bbox_to_image(test_image[0], test_image[1]["boxes"], test_image[1]["labels"], "Test.png")


if ResNet_available:
    FasterRCNN_Net = detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
else 
    FasterRCNN_Net = detection.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
in_features = FasterRCNN_Net.roi_heads.box_predictor.cls_score.in_features
FasterRCNN_Net.roi_heads.box_predictor = FastRCNNPredictor(in_features, 26)
initepoch = 0
if os.path.exists(weight_path):
    checkpoint = torch.load(weight_path)
    FasterRCNN_Net.load_state_dict(checkpoint['model_state_dict'])
    initepoch = checkpoint['epoch']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
FasterRCNN_Net.to(device)
print(FasterRCNN_Net)

# construct an optimizer
params = [p for p in FasterRCNN_Net.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0003,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
# test(test_loader, FasterRCNN_Net, device)
for epoch in range(initepoch, 50):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(FasterRCNN_Net, optimizer, train_loader, device, epoch, print_freq=50)
    torch.save({'epoch': epoch,
                'model_state_dict': FasterRCNN_Net.state_dict()
                }, weight_path)
    # update the learning rate
    lr_scheduler.step()

    if epoch % 5 == 4:
        test(test_loader, FasterRCNN_Net, device, epoch, ResNet_available)
