import os,re,cv2,random,torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import torch.nn.functional as F
import scipy.fftpack as fp
from torchinfo import summary
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

        # 將 video 指針設定到隨機選擇的 frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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


        video_name = self.video_paths[idx]
        flow_path = os.path.join(self.flow_root_dir + '/' + list(self.class_map.keys())[label] + '/' + video_name.split('.')[0])
        video_num=re.findall(r'\d+', flow_path.split('/')[3])[0]
        vis_num = self.csv.loc[self.csv['video'] == int(video_num)].iloc[:, 1]
        im_gray = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)
        stream2 = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
        im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0), axis=1)
        freq = im2freq(frame_cropped)
        shifted_freq = fp.fftshift(freq)
        r=30
        rows, cols,_ = frame_cropped.shape
        crow, ccol = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]
        distance = np.sqrt((X - ccol) ** 2 + (Y - crow) ** 2)
        mask = distance >= r

        mask = mask[:, :, np.newaxis]
        filtered_shifted_freq = shifted_freq * mask
        filtered_freq = fp.ifftshift(filtered_shifted_freq)
        freq2im = lambda data: fp.irfft(fp.irfft(data, axis=1), axis=0)
        filtered_image = freq2im(filtered_freq)

        if self.transform:
            frame_cropped=self.transform(frame_cropped)
            stream2=self.transform(stream2)
            filtered_image=self.transform(filtered_image)
        filtered_image = filtered_image.clone().detach().float()

        return filtered_image, stream2,frame_cropped,label, torch.tensor(float(vis_num.iloc[0]),dtype=torch.float)

rgb_root_dir='./video_class'
flow_root_dir='./flow1'
sift_root_dir='./SIFT'

transform = transforms.Compose([
    transforms.ToTensor()
])

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



class Block1(nn.Module):
    def __init__(self,in_channels):
        super(Block1, self).__init__()
        #block1 ,FOR SIFT NET
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    def forward(self, x):

        x=self.conv1_1(x)
        x=self.conv1_2(x)
        x=self.pool1(x)
        return x


class Block2(nn.Module):
    def __init__(self):
        super(Block2, self).__init__()
        # block2, for flow net
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x0):
        x = self.conv2_1(x0)
        x = self.conv2_2(x)
        x = self.pool2(x)
        return x


class Block3(nn.Module):
    def __init__(self):
        super(Block3, self).__init__()
        # block3,for rgb net
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        return x


class VisNet(nn.Module):
    def __init__(self,sift_channels,flow_channels,rgb_channels,num_classes):
        super(VisNet, self).__init__()
        self.stream1_b1 = Block1(sift_channels)
        self.stream2_b1 = Block1(flow_channels)
        self.stream3_b1 = Block1(rgb_channels)
        self.stream1_b2 = Block2()
        self.stream2_b2 = Block2()
        self.stream3_b2 = Block2()
        self.stream1_b3 = Block3()
        self.stream2_b3 = Block3()
        self.stream3_b3 = Block3()

        self.avg_poo1= nn.AdaptiveAvgPool2d(1)
        self.avg_poo2=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(256,128)
        self.classification_branch = nn.Linear(128, num_classes)
        self.regression_branch = nn.Linear(128, 1)


    def forward(self,x1,x2,x3):
        #x {stream}_{block}
        #stage 1 FOR SIFT NET
        x1_1= self.stream1_b1(x1)
        x2_1= self.stream2_b1(x2)
        x3_1= F.relu(self.stream3_b1(x3))
        x23_1=x2_1+x3_1
        x23_1_=F.interpolate(x23_1,size=x1_1.shape[2:])
        x1_1=x23_1_+x1_1

        #stage 2 for flow net
        x1_2 = self.stream1_b2(x1_1)
        x2_2 = self.stream2_b2(x2_1)
        x3_2 = F.relu(self.stream3_b2(x3_1))
        x23_2=x2_2+x3_2
        x23_2_=F.interpolate(x23_2,size=x1_2.shape[2:])
        x1_2=x23_2_+x1_2


        #stage 3 for RGB
        x1_3=self.stream1_b3(x1_2)
        x2_3=self.stream2_b3(x2_2)
        x3_3=F.relu(self.stream3_b3(x3_2))
        x23_3=x2_3+x3_3

        x1_3=self.avg_poo1(x1_3).view(x1_3.shape[0],-1)
        x23_3=self.avg_poo2(x23_3).view(x23_3.shape[0],-1)

        x_new=x1_3+x23_3
        x_new=F.relu(self.fc(x_new))
        reg=F.sigmoid(self.regression_branch(x_new))
        logit=self.classification_branch(x_new)

        return  logit,reg

def train(model, train_loader, criterion,criterion2,optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs1,inputs2,inputs3, labels, vis_num in train_loader:
        inputs1,inputs2,inputs3, labels, vis_num = inputs1.to(device),inputs2.to(device),inputs3.to(device), labels.to(device),vis_num.to(device)

        optimizer.zero_grad()

        outputs1,outputs2 = model(inputs3,inputs2,inputs1)
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

            outputs1, outputs2 = model(inputs3,inputs2,inputs1)
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
model = VisNet(sift_channels=3,flow_channels=3,rgb_channels=3,num_classes=3)
# input_shape1 = (5, 3, 200, 600)
# input_shape2 = (5, 3, 200, 600)
# input_shape3 = (5, 3, 43,  128)
# summary(model, input_data=(torch.randn(input_shape1), torch.randn(input_shape2), torch.randn(input_shape3)))
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

num_epochs = 120
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
        save_checkpoint(model, optimizer, num_epochs, test_loss, best_model_path)
        #torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with accuracy: {best_accuracy:.4f} at epoch {best_epoch + 1}")


