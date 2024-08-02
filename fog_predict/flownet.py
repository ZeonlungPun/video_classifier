import os,re,random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import torch.nn.functional as F

num_frames=8
class OpticalFlowDataset(Dataset):
    def __init__(self, flow_root_dir, transform=None):
        self.flow_root_dir = flow_root_dir
        self.transform = transform
        self.flow_paths = []
        self.labels = []
        self.class_map = {}
        self.csv=pd.read_csv('./output.csv')
        scaler = MinMaxScaler()
        self.csv.iloc[:,1]=scaler.fit_transform(np.array(self.csv.iloc[:,1]).reshape((-1,1)))
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
        while True:
            label = self.labels[idx]
            flow_path = self.flow_paths[idx]
            flow_path=os.path.join('./flow'+'/'+list(self.class_map.keys())[label]+'/'+flow_path)
            sub_flow_path_list=os.listdir(flow_path)
            sub_flow_path_list = sorted(sub_flow_path_list, key=lambda x: int(re.findall(r'\d+', x)[0]))
            video_name=re.findall(r'\d+',flow_path.split('/')[3])[0]
            vis_num=self.csv.loc[self.csv['video']==int(video_name)].iloc[:,1]

            # Check if there are 20 files
            if len(sub_flow_path_list) < num_frames:
                # Randomly select a new index
                idx = np.random.randint(0, len(self.labels))
                continue

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
            random_frame = np.random.permutation(flows.shape[2])[:num_frames]
            flows=flows[:,:,random_frame]



            if self.transform:
                flows = self.transform(flows)

            return flows, label, torch.tensor(float(vis_num.iloc[0]),dtype=torch.float)
class ChannelAttention(nn.Module):  # Channel Attention Module
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

class CustomBranch(nn.Module):
    def __init__(self, num_classes, in_channel, branch_name):
        super(CustomBranch, self).__init__()
        self.branch_name=branch_name
        # 加載選擇的backbone
        if self.branch_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            # 調整第一層的輸入channel
            self.backbone.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=2, padding=3, bias=False)
            num_ftrs = self.backbone.fc.in_features
            self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        elif self.branch_name == 'ShuffleNetV2':
            self.backbone = models.shufflenet_v2_x0_5(pretrained=True)
            self.backbone.conv1[0]= nn.Conv2d(in_channel, 24, kernel_size=3, stride=2, padding=1, bias=False)
            # 添加AdaptiveAvgPool2d以使輸出形狀固定
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            num_ftrs = self.backbone.fc.in_features
            self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        elif self.branch_name == 'MobileNetV2':
            self.backbone = models.mobilenet_v2(pretrained=True)
            self.backbone.features[0][0] = nn.Conv2d(in_channel, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            num_ftrs = self.backbone.classifier[1].in_features
            # 保存特徵提取部分
            self.feature_extractor = nn.Sequential(*list(self.backbone.features))

        elif self.branch_name == 'VGG16':
            self.backbone = models.vgg16(pretrained=False)
            self.backbone.features[0] = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1)
            num_ftrs = self.backbone.classifier[0].in_features
            # 保存特徵提取部分，直到自適應平均池化層之前
            self.feature_extractor = nn.Sequential(*list(self.backbone.features), nn.AdaptiveAvgPool2d((7, 7)))
        elif self.branch_name == 'xception':
            self.backbone = pretrainedmodels.__dict__['xception'](pretrained=None)
            self.backbone.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, stride=2, padding=0, bias=False)
            num_ftrs = self.backbone.last_linear.in_features
            # 保存特徵提取部分，直到最後一層的全連接層之前
            self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.branch_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=False)
            self.backbone.features.conv0 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.backbone.classifier.in_features
            self.feature_extractor = nn.Sequential(*list(self.backbone.features), nn.ReLU(inplace=True),
                                                   nn.AdaptiveAvgPool2d((1, 1)))
        elif self.branch_name == 'alexnet':
            self.backbone = models.alexnet(pretrained=False)
            self.backbone.features[0] = nn.Conv2d(in_channel, 64, kernel_size=11, stride=4, padding=2)
            self.feature_extractor = nn.Sequential(*list(self.backbone.features), nn.AdaptiveAvgPool2d((1, 1)))
            num_ftrs = 256

        else:
            raise ValueError("Unsupported backbone model.")

        self.classification_branch = nn.Linear(num_ftrs, num_classes)
        self.regression_branch = nn.Linear(num_ftrs, 1)

    def forward(self, x):

        # Feature extraction
        x = self.feature_extractor(x)


        if self.branch_name == 'ShuffleNetV2' or self.branch_name == 'MobileNetV2' or self.branch_name =='xception' :
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Classification output
        class_output = self.classification_branch(x)

        # Regression output (sigmoid for probability-like output)
        reg_output = F.sigmoid(self.regression_branch(x))

        return class_output, reg_output

# 设置光流数据目录
flow_root_dir = './flow'
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = OpticalFlowDataset(flow_root_dir, transform=transform)
train_size = int(0.7 * len(dataset))
val_size= int(0.15*len(dataset))
test_size = len(dataset) - train_size-val_size
train_dataset, test_dataset,val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size,val_size])

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
val_loader= DataLoader(val_dataset,batch_size=5,shuffle=False)

# 打印类别映射和数据集大小
print(f"Class map: {dataset.class_map}")
print(f"Total number of samples: {len(dataset)}")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

for inputs, labels,vis_num in train_loader:
    print(inputs.shape)
    print(labels)
    print(vis_num)
    break


def train(model, train_loader, criterion,criterion2,optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels, vis_num in train_loader:
        inputs, labels, vis_num = inputs.to(device), labels.to(device),vis_num.to(device)

        optimizer.zero_grad()

        outputs1,outputs2 = model(inputs)
        loss1 = criterion(outputs1, labels)
        loss2 = criterion2(outputs2.reshape((-1,1)),vis_num.reshape((-1,1)))

        loss=loss1+loss2
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

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
        for inputs, labels, vis_num in test_loader:
            inputs, labels, vis_num = inputs.to(device), labels.to(device),vis_num.to(device)

            outputs1, outputs2 = model(inputs)
            loss1 = criterion(outputs1, labels)
            loss2 = criterion2(outputs2.reshape((-1,1)), vis_num.reshape((-1,1)))

            loss = loss1 + loss2
            running_loss += loss.item() * inputs.size(0)
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

# 創建模型實例
model = CustomBranch(num_classes=3,branch_name='densenet121',in_channel=num_frames)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
criterion2 = nn.HuberLoss()

# lr=0.0001
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

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
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with accuracy: {best_accuracy:.4f} at epoch {best_epoch + 1}")


