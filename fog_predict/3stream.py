import os,re,cv2,random,torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import torch.nn.functional as F
class ThreeStreamDataset(Dataset):
    def __init__(self, rgb_root_dir,flow_root_dir,sift_root_dir,new_size,transform=None):
        self.rgb_root_dir=rgb_root_dir
        self.flow_root_dir=flow_root_dir
        self.sift_root_dir=sift_root_dir
        self.transform=transform
        self.new_size=new_size
        self.video_paths = []
        self.labels = []
        self.class_map = {}
        self.csv = pd.read_csv('./output.csv')
        scaler = MinMaxScaler()
        self.csv.iloc[:, 1] = scaler.fit_transform(np.array(self.csv.iloc[:, 1]).reshape((-1, 1)))


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

        label = self.labels[idx]
        path = self.video_paths[idx]
        rgb_path = os.path.join(self.rgb_root_dir + '/' + list(self.class_map.keys())[label] + '/' + path)
        cap = cv2.VideoCapture(rgb_path)
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

        video_name = self.video_paths[idx]
        flow_path = os.path.join(self.flow_root_dir + '/' + list(self.class_map.keys())[label] + '/' + video_name.split('.')[0])
        sub_flow_path_list = os.listdir(flow_path)
        sub_flow_path_list = sorted(sub_flow_path_list, key=lambda x: int(re.findall(r'\d+', x)[0]))
        video_num=re.findall(r'\d+', flow_path.split('/')[3])[0]
        vis_num = self.csv.loc[self.csv['video'] == int(video_num)].iloc[:, 1]

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
        #任意選擇兩幀光流
        random_frame = random.randint(0, flows.shape[2] - 2)
        flows = flows[:, :, random_frame:(random_frame + 2)]
        if self.transform:
            flows = self.transform(flows)

        sift_list=[]
        sift_path=os.path.join(self.sift_root_dir + '/' + list(self.class_map.keys())[label] + '/' + video_name.split('.')[0])
        sift_path_list=os.listdir(sift_path)
        for sub_sift_path in sift_path_list:
            final_sift_path=os.path.join(sift_path,sub_sift_path)
            sift_=np.expand_dims(np.load(final_sift_path),2)
            sift_list.append(sift_)
        sift_list= np.concatenate(sift_list, axis=-1)
        random_frame = np.random.permutation(sift_list.shape[2])[:3]
        sift_list=sift_list[:,:,random_frame]
        if self.transform:
            sift_list = self.transform(sift_list)
#三個輸入，兩個輸出： 隨機抽取的1幀，隨機抽取的2幀光流，3幀的SIFT descriptor; 分類類別，能見度值
        return frame_cropped ,flows,sift_list, label,torch.tensor(float(vis_num.iloc[0]),dtype=torch.float)

rgb_root_dir='./video_class'
flow_root_dir='./flow1'
sift_root_dir='./SIFT'

transform = transforms.Compose([
    transforms.ToTensor()
])
#new_size=(height,wdith)
dataset = ThreeStreamDataset(rgb_root_dir,flow_root_dir,sift_root_dir,new_size=(200,600), transform=transform)
train_size = int(0.7 * len(dataset))
val_size= int(0.15*len(dataset))
test_size = len(dataset) - train_size-val_size
train_dataset, test_dataset,val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size,val_size])

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
val_loader= DataLoader(val_dataset,batch_size=5,shuffle=False)

for frame,flow,sift, labels,vis_num  in train_loader:
    print(frame.shape)
    print(flow.shape)
    print(sift.shape)
    print(labels)
    print(vis_num)
    break


class CustomBranch(nn.Module):
    def __init__(self, num_classes, in_channel):
        super(CustomBranch, self).__init__()

        # 加載選擇的backbone
        self.backbone = models.shufflenet_v2_x0_5(pretrained=True)

        self.backbone.conv1[0] = nn.Conv2d(in_channel, 24, kernel_size=3, stride=2, padding=1, bias=False)

        # 添加AdaptiveAvgPool2d以使輸出形狀固定
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs =  1024
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])

        self.classification_branch = nn.Linear(num_ftrs, num_classes)
        self.regression_branch = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        # Feature extraction
        x_mid = self.feature_extractor(x)
        x = self.avgpool(x_mid)
        x = x.view(x.size(0), -1)

        return x, x_mid  # 返回最終特徵和中間特徵

class ThreeStream(nn.Module):
    def __init__(self, num_classes, optical_channels, sift_channels):
        super(ThreeStream, self).__init__()
        self.SpatialModel = CustomBranch(num_classes=num_classes, in_channel=3)
        self.OpticalModel = CustomBranch(num_classes=num_classes, in_channel=optical_channels)
        self.SiftModel = CustomBranch(num_classes=num_classes, in_channel=sift_channels)

        self.class_weights = nn.Parameter(torch.ones(3))
        self.reg_weights = nn.Parameter(torch.ones(3))

        # Ensure the same number of output channels for all branches
        self.conv_spatial_to_optical = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv_optical_to_spatial = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv_sift_to_spatial = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv_sift_to_optical = nn.Conv2d(1024, 1024, kernel_size=1)

    def forward(self, x1, x2, x3):
        features1, intermediate_features1_ = self.OpticalModel(x2)
        features2, intermediate_features2_ = self.SpatialModel(x1)
        features3, intermediate_features3_ = self.SiftModel(x3)

        # Adding horizontal connections in convolutional layers
        #Spatial TO Optical
        intermediate_features1 = intermediate_features1_ + self.conv_spatial_to_optical(intermediate_features2_)
        #optical to spatial
        intermediate_features2_1 = intermediate_features2_ + self.conv_optical_to_spatial(intermediate_features1_)
        #sift to optical
        intermediate_features3_ = F.interpolate(intermediate_features3_, size=intermediate_features1_.shape[2:])
        intermediate_features3 = intermediate_features3_ + self.conv_sift_to_optical(intermediate_features3_)
        #sift to spatial
        intermediate_features3_ = F.interpolate(intermediate_features3_, size=intermediate_features2_.shape[2:])
        intermediate_features2_2 = intermediate_features3_ + self.conv_sift_to_spatial(intermediate_features3_)

        # Pass through the rest of the feature extractor
        final_features1 = self.OpticalModel.avgpool(intermediate_features1+intermediate_features3_)
        final_features1 = final_features1.view(final_features1.size(0), -1)+features1
        final_features2 = self.SpatialModel.avgpool(intermediate_features2_1+intermediate_features2_2)
        final_features2 = final_features2.view(final_features2.size(0), -1)+features2


        # Classification and regression branches
        class_output1 = self.OpticalModel.classification_branch(features1)
        class_output2 = self.SpatialModel.classification_branch(features2)
        class_output3 = self.SiftModel.classification_branch(features3)
        reg_output1 = F.sigmoid(self.OpticalModel.regression_branch(final_features1))
        reg_output2 = F.sigmoid(self.SpatialModel.regression_branch(final_features2))
        reg_output3 = F.sigmoid(self.SiftModel.regression_branch(features3))

        # 使用softmax来正规化权重
        weights1 = nn.functional.softmax(self.class_weights, dim=0)
        weights2 = nn.functional.softmax(self.reg_weights, dim=0)

        # 将logit1, logit2和logit3根据正规化后的权重进行组合
        logit = weights1[0] * class_output1 + weights1[1] * class_output2 + weights1[2] * class_output3
        reg = weights2[0] * reg_output1 + weights2[1] * reg_output2 + weights2[2] * reg_output3

        return logit, reg
def train(model, train_loader, criterion,criterion2,optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs1,inputs2,inputs3, labels, vis_num in train_loader:
        inputs1,inputs2,inputs3, labels, vis_num = inputs1.to(device),inputs2.to(device),inputs3.to(device), labels.to(device),vis_num.to(device)

        optimizer.zero_grad()

        outputs1,outputs2 = model(inputs1,inputs2,inputs3)
        loss1 = criterion(outputs1, labels)
        loss2 = criterion2(outputs2.reshape((-1,1)),vis_num.reshape((-1,1)))

        loss=loss1+loss2
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs1.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def test(model, test_loader, criterion,criterion2, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_outputs2 = []
    with torch.no_grad():
        for inputs1,inputs2,inputs3, labels, vis_num in test_loader:
            inputs1,inputs2,inputs3, labels, vis_num = inputs1.to(device),inputs2.to(device),inputs3.to(device), labels.to(device),vis_num.to(device)

            outputs1, outputs2 = model(inputs1,inputs2,inputs3)
            loss1 = criterion(outputs1, labels)
            loss2 = criterion2(outputs2.reshape((-1,1)), vis_num.reshape((-1,1)))

            loss = loss1 + loss2
            running_loss += loss.item() * inputs1.size(0)
            _, predicted = torch.max(torch.softmax(outputs1,1), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect all true and predicted values for R2 calculation
            all_labels.extend(vis_num.cpu().numpy())
            all_outputs2.extend(outputs2.cpu().numpy())

    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total
    # Calculate R2 score
    r2 = r2_score(all_labels, all_outputs2)
    return epoch_loss, accuracy,r2

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

# 創建模型實例
model = ThreeStream(num_classes=3,optical_channels=2,sift_channels=3)
pretrained_model_path=None
# 計算模型參數量
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



criterion = nn.CrossEntropyLoss()
criterion2 = nn.HuberLoss()

# lr=0.0001
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
if pretrained_model_path is not None:
    model, optimizer, start_epoch, loss = load_checkpoint(model, optimizer, pretrained_model_path)
# 設置追蹤最佳模型的變數
best_accuracy = 0.0
best_epoch = 0
best_model_path = "best_model.pth"

num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion,criterion2, optimizer, device)
    test_loss, test_accuracy,r2 = test(model, test_loader, criterion,criterion2, device)
    # 在每個epoch結束時更新學習率
    #scheduler.step()
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f},r2:{r2:.4f} ")
    if test_accuracy+r2> best_accuracy:
        best_accuracy = test_accuracy+r2
        best_epoch = epoch
        #save_checkpoint(model, optimizer, num_epochs, test_loss, best_model_path)
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with accuracy: {best_accuracy:.4f} at epoch {best_epoch + 1}")
