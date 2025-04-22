import torch,random,joblib,cv2
from torchvision import transforms
from threestream import ThreeStreamDataset,ThreeStream
import numpy as np

scaler = joblib.load('scaler.pkl')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ThreeStream(num_classes=3,optical_channels=2,sift_channels=3)  # <-- 你的模型結構
model.load_state_dict(torch.load('./best_model3stream.pth', map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

# 這裡 new_size 要跟你的訓練時候設定的一樣
dataset = ThreeStreamDataset(
    rgb_root_dir='./video_class',
    flow_root_dir='./flow',
    sift_root_dir='./SIFT',
    new_size=(200, 600),
    transform=transform
)

# 隨便選一個樣本
sample_idx = random.randint(0, len(dataset) - 1)
rgb, flow, sift, label_class,label_vis = dataset[sample_idx]
#dataset.video_paths[sample_idx]
#加一個 batch 維度
rgb = rgb.unsqueeze(0).to(device)
flow = flow.unsqueeze(0).to(device)
sift = sift.unsqueeze(0).to(device)
# def read_sample(rgb_path, flow_path, sift_path, new_size=(200, 600)):
#     # 讀取 RGB 圖像
#     rgb = cv2.imread(rgb_path)
#     rgb = cv2.resize(rgb, new_size)
#     rgb = transform(rgb).unsqueeze(0).to(device)
#
#     # 讀取光流數據
#     flow = np.load(flow_path)
#     flow = np.expand_dims(flow, axis=0)  # 假設 flow 數據需要擴展為3維
#     flow = transform(flow).unsqueeze(0).to(device)
#
#     # 讀取 SIFT 特徵數據
#     sift = np.load(sift_path)
#     sift = np.expand_dims(sift, axis=0)  # 假設 sift 數據需要擴展為3維
#     sift = transform(sift).unsqueeze(0).to(device)
#
#     return rgb, flow, sift
# rgb_path=
# flow_path=
# sift_path=
# rgb, flow, sift = read_sample(rgb_path, flow_path, sift_path)


with torch.no_grad():
    prediction = model(rgb, flow, sift)

vis_pred=prediction[1]
pred_logit=prediction[0]
pred_class=torch.argmax(pred_logit).cpu().numpy()
label_vis=scaler.inverse_transform(label_vis.cpu().numpy().reshape((-1, 1)))
vis_pred=scaler.inverse_transform(vis_pred.cpu().numpy().reshape((-1, 1)))
print("pred:",vis_pred)
print("true:",label_vis)
print('true class:',label_class)
print('predicted class:',pred_class)