import cv2,os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
# 定義數據轉換
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 960)),
    transforms.ToTensor(),
])
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=transform):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        video_path=os.path.join('../input_video',video_path)
        cap = cv2.VideoCapture(video_path)
        frames = []
        flows = []

        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError(f"Error reading video {video_path}")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_count = 0
        while ret:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 計算光流
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,0.5, 3, 15, 3, 5, 1.2, 0)
            prev_gray = gray

            # 將光流轉換為RGB格式
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            if self.transform:
                frame = self.transform(frame)
                rgb_flow = self.transform(rgb_flow)

            frames.append(frame)
            flows.append(rgb_flow)

            frame_count += 1


        cap.release()

        # 將列表轉換為Tensor
        frames = torch.stack(frames, dim=1)
        flows = torch.stack(flows, dim=1)

        return frames, flows, label




if __name__ == '__main__':
    labels_list = pd.read_csv('../output/output.csv').iloc[:, 3]
    dataset = VideoDataset(video_paths=os.listdir('../input_video'), labels=labels_list)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for inputs1,inputs2, labels in dataloader:
        print(inputs1.shape)
        print(inputs2.shape)
        print(labels)
        break