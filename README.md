# Faster R-CNN for ASL letters detection
As the final project of Deep Learning, this is a Faster R-CNN that can locate the American Sign Language(ASL) letters in images and classify the detected letters. The training and testing of this network is based on the American Sign Language Letters from Kaggle.
Due to the limitation of the size of uploaded files, the pretrained weights of the network are not uploaded. However, the trained output are given in the Ouput folder, inluding the general evaluation of network performance and some experimental examples.

# Prerequisites
* Python 3.10
* PyTorch 2.0.0
* PyTorch-cuda 11.7
* CUDA 12.1
* Torchvision 0.15.0
* pycocotools 2.0.6

# Training and Testing
To train the network, please simply run main.py. Since we implemented two different Faster R-CNN networks bases on ResNet and MobileNetv3 respectively, please set the variable "ResNet_available" in main.py to evaluate different networks.
