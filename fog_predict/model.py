import torch
import torch.nn as nn
import torchvision.models as models


class ResNet3D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3D, self).__init__()
        self.resnet3d = models.video.r3d_18(pretrained=True)
        self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet3d(x)


class MC3Model(nn.Module):
    def __init__(self, num_classes):
        super(MC3Model, self).__init__()
        self.mc3 = models.video.mc3_18(pretrained=True)
        self.mc3.fc = nn.Linear(self.mc3.fc.in_features, num_classes)

    def forward(self, x):
        return self.mc3(x)
if __name__ == '__main__':
    #model=VideoRegressModel(IMAGE_HEIGHT=720,IMAGE_WIDTH=1280)
    #model=LRCNModel(IMAGE_HEIGHT=720,IMAGE_WIDTH=1280,num_classes=3)
    #model=Conv3D_LSTM_Model()
    model=ResNet3D(num_classes=4)
    x=torch.randn((4, 3, 6, 720, 1280))
    y=model(x)
    print(len(y))