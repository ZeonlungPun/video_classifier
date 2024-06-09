import os,random,cv2,torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim


class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, slow_frames=20, fast_frames=40):
        self.root_dir = root_dir
        self.transform = transform
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.video_files = []
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for video_file in os.listdir(cls_folder):
                if video_file.endswith('.avi'):
                    self.video_files.append((os.path.join(cls_folder, video_file), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path, label = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        total_frames = len(frames)

        # 確保總幀數足夠
        if total_frames < max(self.slow_frames, self.fast_frames):
            frames = frames * (max(self.slow_frames, self.fast_frames) // total_frames + 1)
            total_frames = len(frames)

        # 隨機選取幀
        slow_indices = sorted(random.sample(range(total_frames), self.slow_frames))
        fast_indices = sorted(random.sample(range(total_frames), self.fast_frames))

        slow_frames = [frames[i] for i in slow_indices]
        fast_frames = [frames[i] for i in fast_indices]

        if self.transform:
            slow_frames = [self.transform(frame) for frame in slow_frames]
            fast_frames = [self.transform(frame) for frame in fast_frames]

        slow_frames = torch.stack(slow_frames).permute(1, 0, 2, 3)  # C, T, H, W
        fast_frames = torch.stack(fast_frames).permute(1, 0, 2, 3)  # C, T, H, W

        return (slow_frames, fast_frames), label

# 定義視頻轉換
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dara_dir='./testvideo'
# 創建訓練和驗證數據集和數據加載器
dataset =VideoDataset(root_dir=dara_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

# 測試數據加載器
for batch_idx, ((slow_data, fast_data), target) in enumerate(train_loader):
    print(slow_data.shape, fast_data.shape, target.shape)
    break


class SlowFastNetwork(nn.Module):
    def __init__(self, num_classes=1000):
        super(SlowFastNetwork, self).__init__()
        # Slow Pathway
        self.slow_pathway = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # Fast Pathway
        self.fast_pathway = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # 橫向連接分支
        self.lateral_connection = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=1, stride=(2,1,1), padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        # 最後分類層
        self.fc = nn.Linear(88, num_classes)

    def forward(self, x):
        slow_input, fast_input = x
        slow_output = self.slow_pathway(slow_input)
        fast_output = self.fast_pathway(fast_input)

        # 橫向連接
        lateral_output = self.lateral_connection(fast_output)

        slow_output = torch.cat([slow_output , lateral_output],dim=1)
        # Global Average Pooling
        slow_output = slow_output.mean([2, 3, 4])
        fast_output = fast_output.mean([2, 3, 4])
        # 拼接slow和fast pathway的輸出
        combined_output = torch.cat([slow_output, fast_output], dim=1)

        return self.fc(combined_output)


# 初始化模型
model = SlowFastNetwork(num_classes=4)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for (inputs1,inputs2), labels in train_loader:
        inputs1,inputs2, labels= inputs1.to(device),inputs2.to(device) ,labels.to(device)

        optimizer.zero_grad()
        outputs= model((inputs1,inputs2))
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
        for (inputs1,inputs2), labels in test_loader:
            inputs1,inputs2, labels= inputs1.to(device),inputs2.to(device) ,labels.to(device)

            outputs= model((inputs1,inputs2))
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs1.size(0)
            _, predicted = torch.max(torch.softmax(outputs,1), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()

# lr=0.0001
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# 設置追蹤最佳模型的變數
best_accuracy = 0.0
best_epoch = 0
best_model_path = "best_model.pth"

num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy= test(model, val_loader, criterion, device)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f} ")
    if test_accuracy> best_accuracy:
        best_accuracy = test_accuracy
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with accuracy: {best_accuracy:.4f} at epoch {best_epoch + 1}")