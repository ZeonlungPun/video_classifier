import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch,cv2,os
from PIL import Image

transform = transforms.Compose([
    transforms.CenterCrop((640, 960)),
    transforms.ToTensor()
])
# 定義自定義數據集類
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=transform):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # 加载视频并进行预处理
        video = self.load_video(video_path)
        if len(video) == 0:
            print(f"Failed to load video: {video_path}")
            return torch.empty(0), label

        if self.transform:
            frames = [self.transform(frame) for frame in video]
        frames = torch.stack(frames, dim=1)  # (C, T, H, W)
        return frames, label

    def load_video(self, video_path):
        video_name=os.path.join('./input_video',video_path)
        cap = cv2.VideoCapture(video_name)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)

        cap.release()
        return frames

if __name__ == '__main__':
    labels_list = pd.read_csv('./output/output.csv').iloc[:,3]
    train_dataset = VideoDataset(video_paths=os.listdir('./input_video'), labels=labels_list, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for inputs, labels in train_loader:
        print(inputs.shape)
        print(labels)
        break




