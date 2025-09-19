import os, re, torch, cv2
from PIL import Image
import numpy as np
from torchvision.models.video import mc3_18
import torch.nn as nn
from video_classifier_dataset import uniform_sample
img_series_path = '/home/zonekey/project/3d_train_test/test/movement/t2029.44-p0'
new_size = (224,112)
frame_num = 10
img_list = []

if os.path.isdir(img_series_path):  # 如果是圖片文件夾
    all_img_name = [f for f in os.listdir(img_series_path) if f.endswith('.jpg')]
    #all_img_name = sorted(all_img_name, key=lambda x: float(re.findall(r'(\d+(?:\.\d+)?)\.png', x)[0]))
    for img_name in all_img_name:
        full_img_path = os.path.join(img_series_path, img_name)
        img_pil = Image.open(full_img_path).convert("RGB").resize(new_size)
        img_array = np.transpose(np.array(img_pil), (2, 0, 1))
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        img_list.append(img_tensor)
else:  # 如果是視頻文件
    cap = cv2.VideoCapture(img_series_path)
    while True:
        ret, cur_frame = cap.read()
        if not ret:
            break
        cur_frame = cv2.resize(cur_frame, new_size)
        cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
        img_array = np.transpose(cur_frame, (2, 0, 1))
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        img_list.append(img_tensor)

# 補齊/裁切幀
# img_num = len(img_list)
# if img_num < frame_num:
#     img_list.extend([img_list[-1].clone() for _ in range(frame_num - img_num)])
# else:
#     img_list = img_list[:frame_num]
img_list =uniform_sample(img_list,  frame_num)
# 構建模型輸入
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
clip_tensor = torch.stack(img_list).permute((1,0,2,3)).unsqueeze(0).to(device)

# 載入模型
class_names =  ['always_sit',  'movement', 'stand_up']
model = mc3_18(pretrained=False)
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('./mc3_action_3class.pth', map_location=device))
model.to(device)
model.eval()

# 推理
with torch.no_grad():
    outputs = model(clip_tensor)
    _, preds = torch.max(outputs, 1)
    preds = preds.item()

print('prediction:', class_names[preds])
