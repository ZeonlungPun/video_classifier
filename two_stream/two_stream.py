import os,re,cv2,random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


class TwoStreamDataset(Dataset):
    def __init__(self, rgb_root_dir,flow_root_dir,new_size,transform=None):
        self.rgb_root_dir=rgb_root_dir
        self.flow_root_dir=flow_root_dir
        self.transform=transform
        self.new_size=new_size
        self.video_paths = []
        self.labels = []
        self.class_map = {}


        class_idx = 0
        for class_name in os.listdir(self.rgb_root_dir):
            class_folder = os.path.join(self.rgb_root_dir, class_name)
            if os.path.isdir(class_folder):
                self.class_map[class_name] = class_idx
                class_idx += 1

                for video_name in os.listdir(class_folder):
                    self.video_paths.append(video_name)
                    self.labels.append(self.class_map[class_name])


    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        while True:
            label = self.labels[idx]
            path = self.video_paths[idx]
            path = os.path.join(self.rgb_root_dir + '/' + list(self.class_map.keys())[label] + '/' + path)
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

            flow_path = self.video_paths[idx]
            flow_path = os.path.join(self.flow_root_dir + '/' + list(self.class_map.keys())[label] + '/' + flow_path.split('.')[0])
            sub_flow_path_list = os.listdir(flow_path)
            sub_flow_path_list = sorted(sub_flow_path_list, key=lambda x: int(re.findall(r'\d+', x)[0]))


            # Check if there are 20 files
            if len(sub_flow_path_list) <= 20:
                # Randomly select a new index
                idx = np.random.randint(0, len(self.labels))
                continue

            flowx, flowy = [], []
            for sub_flow_path in sub_flow_path_list:
                if sub_flow_path[5] == 'x':
                    final_path_x = os.path.join(flow_path, sub_flow_path)
                    flowx_ = np.expand_dims(np.load(final_path_x), 2)
                    flowx.append(flowx_)
                else:
                    final_path_y = os.path.join(flow_path, sub_flow_path)
                    flowy_ = np.expand_dims(np.load(final_path_y), 2)
                    flowy.append(flowy_)
            flowx = np.concatenate(flowx, axis=-1)
            flowy = np.concatenate(flowy, axis=-1)

            flows = np.concatenate([flowx, flowy], axis=2)  # Stack flow x and y

            if self.transform:
                flows = self.transform(flows)

            return frame_cropped ,flows, label

rgb_root_dir='./testvideo'
flow_root_dir='./flow'

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = TwoStreamDataset(rgb_root_dir,flow_root_dir,new_size=(150,250), transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

for frame,flow, labels in train_loader:
    print(frame.shape)
    print(flow.shape)
    print(labels)
    break

class ResNet18Custom(nn.Module):
    def __init__(self, num_classes,in_channel=3):  # 添加回歸輸出的數量
        super(ResNet18Custom, self).__init__()
        # 加載預設的resnet18模型
        self.resnet18 = models.resnet50(pretrained=True)
        # 修改第一層以接受20通道的輸入
        self.resnet18.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=2, padding=3, bias=False)
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

class TwoStream(nn.Module):
    def __init__(self,num_classes,optical_channels):
        super(TwoStream,self).__init__()
        self.SpatialModel = ResNet18Custom(num_classes=num_classes,in_channel=3)
        self.OpticalModel= ResNet18Custom(num_classes=num_classes,in_channel=optical_channels)

    def forward(self, x1,x2):
        logit1=self.OpticalModel(x2)
        logit2=self.SpatialModel(x1)
        logit=(logit1+logit2)/2
        return logit
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs1,inputs2, labels in train_loader:
        inputs1,inputs2, labels= inputs1.to(device),inputs2.to(device) ,labels.to(device)

        optimizer.zero_grad()
        outputs= model(inputs1,inputs2)
        loss = criterion(outputs, labels)
        loss=loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs1.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs1,inputs2, labels in test_loader:
            inputs1,inputs2, labels= inputs1.to(device),inputs2.to(device) ,labels.to(device)

            outputs= model(inputs1,inputs2)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs1.size(0)
            _, predicted = torch.max(torch.softmax(outputs,1), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


# 創建模型實例
model = TwoStream(num_classes=4,optical_channels=40)

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

num_epochs = 100
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