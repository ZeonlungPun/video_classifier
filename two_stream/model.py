import torch.nn as nn
import torch
from torchvision import models


class TwoStreamNetwork(nn.Module):
    def __init__(self, num_classes):
        super(TwoStreamNetwork, self).__init__()
        self.spatial_model = models.video.r3d_18(pretrained=True)
        self.temporal_model = models.video.r3d_18(pretrained=True)

        # 獲取特徵向量的大小
        spatial_in_features = self.spatial_model.fc.in_features
        temporal_in_features = self.temporal_model.fc.in_features

        # 替換分類層
        self.spatial_model.fc = nn.Identity()
        self.temporal_model.fc = nn.Identity()

        self.fc = nn.Linear(spatial_in_features + temporal_in_features, num_classes)

    def forward(self, spatial_input, temporal_input):
        spatial_features = self.spatial_model(spatial_input)
        temporal_features = self.temporal_model(temporal_input)
        combined_features = torch.cat((spatial_features, temporal_features), dim=1)
        output = self.fc(combined_features)
        return output

if __name__ == '__main__':
    num_classes = 2  # 根據你的數據集調整
    model = TwoStreamNetwork(num_classes)
    x = torch.randn((2, 3, 6, 320, 640))
    y = model(x,x)
    print(len(y))
