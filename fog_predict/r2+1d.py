import torch.nn as nn
import torch.nn.functional as F
import torch,os,re
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torchvision.models.video import r2plus1d_18
from sklearn.metrics import r2_score
import torch.optim as optim

def center_crop(video, new_height, new_width):
    """
    Crop the center of each frame in the video.
    Args:
        video: Tensor of shape (num_frames,height, width,channels).
        new_height: The height of the cropped area.
        new_width: The width of the cropped area.
    
    Returns:
        cropped_video: Tensor of shape (num_frames, new_height, new_width, channels).
    """
    num_frames, height, width, channels = video.shape
    center_x, center_y = width // 2, height // 2
    
    crop_x1 = center_x - new_width // 2
    crop_y1 = center_y - new_height // 2
    crop_x2 = center_x + new_width // 2
    crop_y2 = center_y + new_height // 2
    
    # Crop the center area
    cropped_video = video[:, crop_y1:crop_y2, crop_x1:crop_x2,:]
    
    return cropped_video


"""
目錄結構：
video
├── biking
│   ├── v_biking_01_01.avi
│   ├── v_biking_01_02.avi
│   ├── v_biking_02_01.avi
├── juggle
│   ├── v_juggle_01_01.avi
│   ├── v_juggle_01_02.avi
│   ├── v_juggle_01_03.avi
├── ridding
│   ├── v_riding_01_01.avi
│   ├── v_riding_01_02.avi
│   ├── v_riding_01_03.avi
└── shooting
    ├── v_shooting_01_01.avi
    ├── v_shooting_01_02.avi
    ├── v_shooting_01_03.avi

"""
class VideoDataset(Dataset):
    def __init__(self, root_dir,new_size,  frames_per_clip=16):
        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.samples = []
        self.class_names = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.new_size=new_size
        self.csv = pd.read_csv('./output.csv')
        scaler = MinMaxScaler()
        self.csv.iloc[:, 1] = scaler.fit_transform(np.array(self.csv.iloc[:, 1]).reshape((-1, 1)))

        for cls_name in self.class_names:
            cls_dir = os.path.join(root_dir, cls_name)
            for video_name in os.listdir(cls_dir):
                video_path = os.path.join(cls_dir, video_name)
                self.samples.append((video_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video_name = re.findall(r'\d+', video_path.split('/')[3])[0]
        vis_num = self.csv.loc[self.csv['video'] == int(video_name)].iloc[:, 1]
        
        video, _, info = io.read_video(video_path, pts_unit='sec')
        num_frames = video.shape[0]

        if num_frames >= self.frames_per_clip:
            indices = torch.linspace(0, num_frames - 1, self.frames_per_clip).long()
            video_clip = video[indices]
        else:
            # 如果幀數不足，則補零
            video_clip = torch.zeros((self.frames_per_clip, *video.shape[1:]))
            video_clip[:num_frames] = video
        
        #video_clip :(time, h,w,c) 
        new_height, new_width = self.new_size
        video_clip = center_crop(video_clip, new_height, new_width)  # 改變維度順序進行裁剪
        #轉換為 (C, T, H, W)
        video_clip=video_clip.permute(3,0,1,2).contiguous().float()
       
       
        return video_clip, label,torch.tensor(float(vis_num.iloc[0]),dtype=torch.float)




dataset = VideoDataset(root_dir='./video_class',new_size=(200,600), frames_per_clip=4)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)
print(dataset.class_names)
for inputs, labels,vis_num in train_loader:
    print(inputs.shape)
    print(labels)
    print(vis_num)
    break



# 加載預訓練的 R(2+1)D-18 模型
backbone = r2plus1d_18(weights=None)


# 获取原始全连接层的输入特征数量
in_features =backbone.fc.in_features
# 修改模型的最後一層，使其同時輸出分類和回歸值
num_classes = 4  # 替換為實際的類別數
num_regression_outputs = 1  # 回歸輸出的維度

# 替換模型的全連接層
backbone.fc = nn.Identity()  # 移除預設的全連接層

# 創建新的分類和回歸分支
class r2plus1dModel(nn.Module):
    def __init__(self, backbone, num_classes, num_regression_outputs):
        super(r2plus1dModel, self).__init__()
        self.backbone = backbone
        self.classification_head = nn.Linear(in_features, num_classes)
        self.regression_head = nn.Linear(in_features, num_regression_outputs)

    def forward(self, x):
        features = self.backbone(x)
        class_output = self.classification_head(features)
        regression_output = self.regression_head(features)
        return class_output, regression_output



model = r2plus1dModel(backbone,num_classes,num_regression_outputs)




def train(model, train_loader, criterion,criterion2, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels, vis_num in train_loader:
        inputs, labels, vis_num = inputs.to(device), labels.to(device), vis_num.to(device)

        optimizer.zero_grad()

        outputs1, outputs2 = model(inputs)
        loss1 = criterion(outputs1, labels)
        loss2 = criterion2(outputs2.reshape((-1, 1)), vis_num.reshape((-1, 1)))
        loss = loss1 +  loss2
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def test(model, test_loader, criterion,criterion2,criterion3, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    total_mse=0
    all_labels = []
    all_outputs2 = []
    with torch.no_grad():
        for inputs, labels, vis_num in test_loader:
            inputs, labels, vis_num = inputs.to(device), labels.to(device),vis_num.to(device)

            outputs1, outputs2 = model(inputs)
            loss1 = criterion(outputs1, labels)
            loss2 = criterion2(outputs2.reshape((-1,1)), vis_num.reshape((-1,1)))
            mse= criterion3(outputs2.reshape((-1,1)), vis_num.reshape((-1,1))).item()
            total_mse +=mse
            loss = loss1 + loss2
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(torch.softmax(outputs1,1), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect all true and predicted values for R2 calculation
            all_labels.extend(vis_num.cpu().numpy())
            all_outputs2.extend(outputs2.cpu().numpy())
    ave_mse=total_mse/ len(test_loader.dataset)
    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total
    # Calculate R2 score
    r2 = r2_score(all_labels, all_outputs2)
    return epoch_loss, accuracy,r2,ave_mse



# 計算模型參數量的函數
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# 計算參數量
num_params = count_parameters(model)
print(f"模型的參數量: {num_params}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
criterion2 = nn.HuberLoss()
criterion3 =nn.MSELoss()
# lr=0.0001
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# 設置追蹤最佳模型的變數
best_accuracy = 0.0
best_epoch = 0
best_model_path = "best_model.pth"

num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion,criterion2, optimizer, device)
    test_loss, test_accuracy,r2,ave_mse= test(model, val_loader, criterion,criterion2,criterion3,device)
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f},r2:{r2:.4f},mse:{ave_mse:.4f} ")
    if test_accuracy+r2>= best_accuracy:
        best_accuracy = test_accuracy+r2
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with accuracy: {best_accuracy:.4f} at epoch {best_epoch + 1}")