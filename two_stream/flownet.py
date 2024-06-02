import os,re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

class OpticalFlowDataset(Dataset):
    def __init__(self, flow_root_dir, transform=None):
        self.flow_root_dir = flow_root_dir
        self.transform = transform
        self.flow_paths = []
        self.labels = []
        self.class_map = {}
        class_idx = 0


        for class_name in os.listdir(flow_root_dir):
            class_folder = os.path.join(flow_root_dir, class_name)
            if os.path.isdir(class_folder):
                self.class_map[class_name] = class_idx
                class_idx += 1

                for video_name in os.listdir(class_folder):
                    self.flow_paths.append(video_name)
                    self.labels.append(self.class_map[class_name])

    def __len__(self):
        return len(self.flow_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        flow_path = self.flow_paths[idx]
        flow_path=os.path.join('./flow'+'/'+list(self.class_map.keys())[label]+'/'+flow_path)
        sub_flow_path_list=os.listdir(flow_path)
        sub_flow_path_list = sorted(sub_flow_path_list, key=lambda x: int(re.findall(r'\d+', x)[0]))
        flowx,flowy=[],[]
        for sub_flow_path in sub_flow_path_list:
            if sub_flow_path[5]=='x':
                final_path_x=os.path.join(flow_path,sub_flow_path)
                flowx_=np.expand_dims(np.load(final_path_x),2)
                flowx.append(flowx_)
            else:
                final_path_y = os.path.join(flow_path, sub_flow_path)
                flowy_=np.expand_dims(np.load(final_path_y),2)
                flowy.append(flowy_)
        flowx = np.concatenate(flowx, axis=-1)
        flowy = np.concatenate(flowy, axis=-1)


        flows = np.concatenate([flowx, flowy], axis=2)  # Stack flow x and y



        if self.transform:
            flows = self.transform(flows)

        return flows, label
class ResNet18Custom(nn.Module):
    def __init__(self, num_classes):  # num_classes 根據你的分類任務調整
        super(ResNet18Custom, self).__init__()
        # 加載預設的resnet18模型
        self.resnet18 = models.resnet50(pretrained=True)
        # 修改第一層以接受40通道的輸入
        self.resnet18.conv1 = nn.Conv2d(40, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.resnet18.bn1 = nn.BatchNorm2d(64)
        # 替換分類層以符合目標類別數量
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet18(x)

# 设置光流数据目录
flow_root_dir = './flow'
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = OpticalFlowDataset(flow_root_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=15, shuffle=False)

# 打印类别映射和数据集大小
print(f"Class map: {dataset.class_map}")
print(f"Total number of samples: {len(dataset)}")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")




def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(torch.softmax(outputs,1), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

# 創建模型實例
model = ResNet18Custom(num_classes=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

# 設置追蹤最佳模型的變數
best_accuracy = 0.0
best_epoch = 0
best_model_path = "best_model.pth"

num_epochs = 150
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = test(model, test_loader, criterion, device)
    # 在每個epoch結束時更新學習率
    #scheduler.step()
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with accuracy: {best_accuracy:.4f} at epoch {best_epoch + 1}")