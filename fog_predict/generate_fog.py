import os.path

import torch
import cv2
import numpy as np


def detect_sky_region(dark_channel):
    sky_mask = (dark_channel > 0.9 * dark_channel.max()).astype(np.uint8)
    return sky_mask

def auto_estimate_max_depth(depth_map, image):
    dark_channel = get_dark_channel(image)
    sky_mask = detect_sky_region(dark_channel)

    # 假設天空的距離為1000
    if np.sum(sky_mask) > 0:
        return 1000
    else:
        depth_flatten = depth_map.flatten()
        far_depth = np.percentile(depth_flatten, 95)
        # 實驗系數
        return far_depth * 150

delta = 0.05  # 視覺對比閾值

# ----------------- 模型加載 ----------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
model.to(device)
model.eval()

# ----------------- 圖像處理 ----------------- #
def adjust_brightness_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    v_mean = np.mean(v)
    s_mean = np.mean(s)
    v = np.clip(v * (1 - v_mean / 255), 0, 255).astype(np.uint8)
    s = np.clip(s * (s_mean / 255), 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)

def generate_depth_map(image, model):
    input_tensor = transform(image).to(device)
    with torch.no_grad():
        depth = model(input_tensor)
    return cv2.normalize(depth.squeeze().cpu().numpy(), None, 0, 1, cv2.NORM_MINMAX)

def guided_filter(I, p, radius=90, eps=0.8):
    # Resize depth map (p) to match the size of the input image (I)
    p_resized = cv2.resize(p, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Apply guided filter
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
    mean_p = cv2.boxFilter(p_resized, cv2.CV_64F, (radius, radius))
    corr_I = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))
    corr_Ip = cv2.boxFilter(I * p_resized, cv2.CV_64F, (radius, radius))

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))

    return mean_a * I + mean_b

def calculate_transmission(depth_map, visibility, max_depth):
    beta = -np.log(delta) / visibility
    return np.exp(-beta * depth_map * max_depth).clip(0.1, 1.0)

def get_dark_channel(image, window_size=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    return cv2.erode(np.min(image, axis=2), kernel)

def estimate_atmospheric_light(image, dark_channel):
    sky_mask = (dark_channel > 0.8*dark_channel.max()).astype(bool)
    return np.mean(image[sky_mask], axis=0) if sky_mask.any() else np.mean(image, axis=(0,1))

def save_depth_map(depth_map, filename, depth_save_path):
    depth_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_visual = depth_visual.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
    depth_save_name = os.path.join(depth_save_path , f"{filename}.png")
    cv2.imwrite(depth_save_name, depth_colored)

# ----------------- 主流程 ----------------- #
def generate_fog_process(visibility, img_path, hazy_path, depth_save_path):
    # 1. 讀取圖像
    image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    filename = img_path.split('.')[-2].split('/')[-1]
    adjusted_image = adjust_brightness_saturation(image)

    # 2. 深度估計
    depth_map = generate_depth_map(adjusted_image, model)
    depth_map_norm = depth_map.astype(np.float32)

    # 3. 引導濾波
    smoothed_depth = guided_filter(
        cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255,
        depth_map_norm)
    save_depth_map(depth_map_norm, f"raw_depth_{filename}", depth_save_path)

    # 4. 參數估算
    max_depth = auto_estimate_max_depth(smoothed_depth, adjusted_image)
    transmission = calculate_transmission(smoothed_depth, visibility, max_depth)
    dark_channel = get_dark_channel(adjusted_image)
    A = estimate_atmospheric_light(adjusted_image, dark_channel)

    # 5. 生成霧霾
    foggy_image = (adjusted_image * transmission[..., None] + A * (1 - transmission[..., None])).clip(0, 255).astype(
        np.uint8)

    # 6. 保存結果
    hazy_img_save_path=os.path.join(str(hazy_path),filename+'.png')
    cv2.imwrite(hazy_img_save_path, cv2.cvtColor(foggy_image, cv2.COLOR_RGB2BGR))