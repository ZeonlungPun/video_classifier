import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os, time,cv2,torch
import numpy as np
from torchvision.models.video import r3d_18
import torch.nn as nn
from collections import defaultdict
class VideoTransform:
    """對整個 clip (所有幀) 同步做數據增強"""
    def __init__(self, resize=(224, 224), use_tubemask=True):
        self.resize = resize
        self.train_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomResizedCrop(resize, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])
        # Random Erasing
        self.random_erasing = transforms.RandomErasing(
            p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0
        )
        self.tube_masking = TubeMasking(mask_ratio=0.15, mode="zero")
        self.use_tubemask = use_tubemask

    def __call__(self, clip, mode="train"):
        seed = np.random.randint(99999)
        transformed_clip = []
        transform = self.train_transform if mode == "train" else self.test_transform
        for img in clip:
            random.seed(seed)
            transformed_clip.append(transform(img))
        #   # (T=10, C=3, H=224, W=112)
        clip_tensor = torch.stack(transformed_clip)  # (T, C, H, W)

        if mode == "train":
            # Random Erasing (逐幀)
            for t in range(clip_tensor.shape[0]):
                clip_tensor[t] = self.random_erasing(clip_tensor[t])

            # TubeMasking (整段 clip)
            if self.use_tubemask:
                clip_tensor = clip_tensor.permute(1, 0, 2, 3)  # (C,T,H,W)
                clip_tensor = self.tube_masking(clip_tensor)
                clip_tensor = clip_tensor.permute(1, 0, 2, 3)  # (T,C,H,W)

        return clip_tensor

# ---- Temporal Jittering ----
def temporal_jitter(img_list, num_frames,mode):
    """隨機打亂時間取樣，模擬不同動作快慢"""
    if len(img_list) < num_frames:
        img_list.extend([img_list[-1]] * (num_frames - len(img_list)))
    else:
        if mode == 'train':
            indices = sorted(random.sample(range(len(img_list)), num_frames))
        else:
            #uniform_sample
            indices = np.linspace(0, len(img_list) - 1, num_frames).astype(int)
        img_list = [img_list[i] for i in indices]
    return img_list
# 確保你的 VideoTransform 類和 Temporal Jittering 函數被導入或定義在此腳本中
# 這裡我直接將它們貼出來，以確保完整性
# --- VideoTransform Class ---
class VideoTransform:
    def __init__(self, resize=(224, 112), use_tubemask=False):  # 注意，推斷時不使用 TubeMasking
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])

    def __call__(self, clip):
        transformed_clip = [self.transform(img) for img in clip]
        clip_tensor = torch.stack(transformed_clip)  # (T, C, H, W)
        return clip_tensor


# --- Temporal Jittering Function ---
def temporal_jitter(img_list, num_frames, mode):
    if len(img_list) < num_frames:
        # 不足就補最後一幀
        img_list.extend([img_list[-1]] * (num_frames - len(img_list)))
    else:
        # 推斷時使用均勻採樣
        indices = np.linspace(0, len(img_list) - 1, num_frames).astype(int)
        img_list = [img_list[i] for i in indices]
    return img_list


# --- 推斷相關配置 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['always_sit', 'movement', 'stand_up']
num_frames = 10  # 確保與訓練時的 frames_per_clip 參數一致


def load_model(model_name, num_classes):
    """
    加載模型並載入權重
    """
    if model_name == 'r3d18':
        model = r3d_18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load('./r3d18_action_3class_frozen.pth', map_location=device))
    else:
        raise ValueError("Unsupported model name")

    model.to(device)
    model.eval()
    return model


def inference(video_path, model, class_names):
    """
    對單個視頻路徑進行推斷
    """
    # 1. 讀取圖片序列或視頻文件
    img_list = []
    if os.path.isdir(video_path):
        all_img_name = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
        for img_name in all_img_name:
            full_img_path = os.path.join(video_path, img_name)
            img_pil = Image.open(full_img_path).convert("RGB")
            img_list.append(img_pil)
    else:  # 假設是視頻文件
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, cur_frame = cap.read()
            if not ret:
                break
            cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(cur_frame)
            img_list.append(img_pil)

    # 2. 進行時間抖動 (Temporal Jittering)，使用與訓練時相同的模式 'test'
    img_list = temporal_jitter(img_list, num_frames, mode='test')

    # 3. 進行視頻數據轉換 (VideoTransform)，使用與測試時相同的邏輯
    video_transform = VideoTransform(resize=(224, 112))
    clip_tensor = video_transform(img_list)  # (T, C, H, W)

    # 4. 轉換 Tensor 格式以符合模型輸入 (C, T, H, W)
    clip_tensor = clip_tensor.permute((1, 0, 2, 3))

    # 5. 擴展維度為 (B=1, C, T, H, W)
    inputs = clip_tensor.unsqueeze(0).to(device)

    # 6. 推斷
    with torch.no_grad():
        # 標準化（與訓練時的 DataLoader 邏輯一致）
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return class_names[pred], probs.cpu().numpy()[0]


def evaluate_all_per_class(test_base_path, model, class_names):
    """
    評估測試集上所有類別的準確率，並顯示每個類別的結果。

    Args:
        test_base_path (str): 測試集根目錄，例如 '/home/zonekey/project/3d_train_test/test'
        model: 已加載的模型
        class_names: 類別名稱列表
    """
    # 使用 defaultdict 方便地追蹤每個類別的計數
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # 遍歷測試集根目錄下的所有類別目錄
    for class_name in class_names:
        class_dir = os.path.join(test_base_path, class_name)

        # 遍歷該類別目錄下的所有視頻樣本
        sample_path_list = os.listdir(class_dir)

        for video_dir in sample_path_list:
            # 獲取單個樣本的完整路徑
            full_video_path = os.path.join(class_dir, video_dir)

            # 進行推斷
            try:
                pred_label_name, _ = inference(full_video_path, model, class_names)

                # 取得真實標籤
                true_label_name = class_name

                # 更新計數
                class_total[true_label_name] += 1
                if pred_label_name == true_label_name:
                    class_correct[true_label_name] += 1
            except Exception as e:
                print(f"Skipping {full_video_path} due to error: {e}")

    print("\n--- 📊 類別準確率 (Per-Class Accuracy) ---")

    # 計算並打印每個類別的準確率
    total_correct = 0
    total_count = 0
    for class_name in class_names:
        correct = class_correct[class_name]
        total = class_total[class_name]
        accuracy = correct / total if total > 0 else 0
        print(f"  ✅ {class_name}: {accuracy:.4f} ({correct}/{total})")

        total_correct += correct
        total_count += total

    # 計算並打印總體準確率
    overall_accuracy = total_correct / total_count if total_count > 0 else 0
    print(f"\n--- ✅ 總體準確率 (Overall Accuracy): {overall_accuracy:.4f} ---")


if __name__ == '__main__':
    model = load_model('r3d18',num_classes=3)

    video_path = "/home/zonekey/project/3d_train_test/test/stand_up/t188.07-p0"
    label, probs = inference(video_path, model, class_names)

    print("Prediction:", label)
    print("Probabilities:", dict(zip(class_names, probs)))

    # 使用正確的 evaluate_all 函數來評估整個測試集
    test_path = '/home/zonekey/project/3d_train_test/test'
    #evaluate_all_per_class(test_path, model, class_names)