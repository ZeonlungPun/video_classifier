import os,cv2,random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


class SpatialDataset(Dataset):
    def __init__(self, root_dir,new_size,transform=None):
        self.root_dir=root_dir
        self.transform=transform
        self.new_size=new_size
        self.video_paths = []
        self.labels = []
        self.class_map = {}

        class_idx = 0
        for class_name in os.listdir(self.root_dir):
            class_folder = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_folder):
                self.class_map[class_name] = class_idx
                class_idx += 1

                for video_name in os.listdir(class_folder):
                    self.video_paths.append(video_name)
                    self.labels.append(self.class_map[class_name])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        path = self.video_paths[idx]
        path = os.path.join(self.root_dir + '/' + list(self.class_map.keys())[label] + '/' + path)
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 隨機選擇一個 frame
        random_frame = random.randint(0, total_frames - 1)

        # 將 video 指針設定到隨機選擇的 frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)

        # 讀取 frame
        ret, frame = cap.read()
        height, width, _ = frame.shape
        new_height, new_width = self.new_size
        center_x, center_y = width // 2, height // 2
        crop_x1 = center_x - new_width // 2
        crop_y1 = center_y - new_height // 2
        crop_x2 = center_x + new_width // 2
        crop_y2 = center_y + new_height // 2
        frame_cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if self.transform:
            frame_cropped=self.transform(frame_cropped)
        return frame_cropped ,label

root_dir='./video_class'
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = SpatialDataset(root_dir,new_size=(100,150), transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

for inputs, labels in train_loader:
    print(inputs.shape)
    print(labels)
    break

class ResNet18Custom(nn.Module):
    def __init__(self, num_classes):  # 添加回歸輸出的數量
        super(ResNet18Custom, self).__init__()
        # 加載預設的resnet18模型
        self.resnet18 = models.resnet50(pretrained=True)
        # 修改第一層以接受20通道的輸入
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3, bias=False)
        # self.resnet18.bn1 = nn.BatchNorm2d(64)
        # 替換分類層以符合目標類別數量
        num_ftrs = self.resnet18.fc.in_features

        # 保存特徵提取部分
        self.feature_extractor = nn.Sequential(*list(self.resnet18.children())[:-1])

        # 定義分類分支
        self.classification_branch = nn.Linear(num_ftrs, num_classes)



    def forward(self, x):
        # 提取特徵
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)

        # 分類輸出
        class_output = self.classification_branch(x)

        return class_output

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels= inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs= model(inputs)
        loss = criterion(outputs, labels)
        loss=loss
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
            inputs, labels= inputs.to(device), labels.to(device)

            outputs= model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(torch.softmax(outputs,1), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


# 創建模型實例
model = ResNet18Custom(num_classes=3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()

# lr=0.0001
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

# 設置追蹤最佳模型的變數
best_accuracy = 0.0
best_epoch = 0
best_model_path = "best_model.pth"

num_epochs = 300
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy= test(model, test_loader, criterion, device)
    # 在每個epoch結束時更新學習率
    #scheduler.step()
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f} ")
    if test_accuracy> best_accuracy:
        best_accuracy = test_accuracy
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with accuracy: {best_accuracy:.4f} at epoch {best_epoch + 1}")