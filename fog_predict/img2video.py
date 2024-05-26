import os
import cv2

# 設置來源文件夾和目標文件夾
source_dir = '/home/punzeonlung/fog_paper/input'
output_video_dir = '/home/punzeonlung/fog_paper/input_video'

# 創建目標文件夾，如果不存在的話
os.makedirs(output_video_dir, exist_ok=True)

# 設定影片參數
fps = 1  # 幀率，每秒1張圖片
frame_size = (1280, 720)  # 設定影片的大小（寬度, 高度），需要根據實際圖片大小調整

# 獲取所有子文件夾
subfolders = [f.path for f in os.scandir(source_dir) if f.is_dir()]

for subfolder in subfolders:
    # 獲取子文件夾名稱
    subfolder_name = os.path.basename(subfolder)

    # 獲取子文件夾中的所有圖片
    image_files = sorted(
        [os.path.join(subfolder, f) for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))])

    # 構建影片的文件名
    output_video_path = os.path.join(output_video_dir, f'{subfolder_name}.avi')

    # 初始化視頻寫入器
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, frame_size)

    for image_file in image_files:
        img = cv2.imread(image_file)
        # 確保圖片大小與影片大小一致
        img = cv2.resize(img, frame_size)
        out.write(img)

    # 釋放視頻寫入器
    out.release()

print("圖片已成功轉換為影片。")
