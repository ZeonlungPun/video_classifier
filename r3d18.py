import os,copy,torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io
from torchvision.models.video import r3d_18
class CustomVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=16):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.samples = []
        self.class_names = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        for cls_name in self.class_names:
            cls_dir = os.path.join(root_dir, cls_name)
            for video_name in os.listdir(cls_dir):
                video_path = os.path.join(cls_dir, video_name)
                self.samples.append((video_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video, _, info = io.read_video(video_path, pts_unit='sec')
        num_frames = video.shape[0]

        if num_frames >= self.frames_per_clip:
            start_idx = torch.randint(0, num_frames - self.frames_per_clip + 1, (1,)).item()
            video_clip = video[start_idx:start_idx + self.frames_per_clip]
        else:
            video_clip = torch.zeros((self.frames_per_clip, *video.shape[1:]))
            video_clip[:num_frames] = video

        if self.transform:
            video_clip = self.transform(video_clip.permute(0, 3, 1, 2).float())  # 改變維度順序
            video_clip = video_clip.permute(1, 0, 2, 3)  # 恢復維度順序，將 (T, C, H, W) 轉換為 (C, T, H, W)

        return video_clip, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.ToTensor()
])

dataset = CustomVideoDataset(root_dir='./testvideo', transform=transform, frames_per_clip=8)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
print(dataset.class_names)


model = r3d_18(pretrained=True)
num_classes = 4  # 替換為你的類別數
model.fc = nn.Linear(model.fc.in_features, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



num_epochs = 25
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
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

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(f'Validation Accuracy: {val_acc:.4f}')

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

# 加載最佳模型權重
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'best_r3d18_model.pth')
print(f'Best Validation Accuracy: {best_acc:.4f}')

# 新視頻樣本推斷
def predict_new_video(model, video_path, transform, frames_per_clip=8):
    model.eval()
    video, _, info = io.read_video(video_path, pts_unit='sec')
    num_frames = video.shape[0]

    if num_frames >= frames_per_clip:
        start_idx = torch.randint(0, num_frames - frames_per_clip + 1, (1,)).item()
        video_clip = video[start_idx:start_idx + frames_per_clip]
    else:
        video_clip = torch.zeros((frames_per_clip, *video.shape[1:]))
        video_clip[:num_frames] = video

    if transform:
        video_clip = transform(video_clip.permute(0, 3, 1, 2).float())  # 改變維度順序
        video_clip = video_clip.permute(1, 0, 2, 3)  # 恢復維度順序，將 (T, C, H, W) 轉換為 (C, T, H, W)

    video_clip = video_clip.unsqueeze(0).to(device)  # 增加 batch 維度
    outputs = model(video_clip)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# 使用最佳模型進行新視頻推斷
model.load_state_dict(torch.load('best_r3d18_model.pth'))

new_video_path = '/home/kingargroo/fog/testvideo/biking/v_biking_01_01.avi'  # 替換為新視頻的路徑
predicted_class_idx = predict_new_video(model, new_video_path, transform)
print(predicted_class_idx)
predicted_class_name = dataset.class_names[predicted_class_idx]

print(f'The predicted class for the new video is: {predicted_class_name}')