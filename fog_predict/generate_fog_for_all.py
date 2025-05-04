import os,csv
from generate_fog import generate_fog_process
from pathlib import Path
import numpy as np
from tqdm import tqdm


# ----------------- 輸入路徑設定 ----------------- #
main_path = './video_f'
img_main_path = Path(main_path + '/img')
depth_path = Path(main_path + '/depth')
hazy_path = Path(main_path + '/hazy')

os.makedirs(hazy_path, exist_ok=True)
os.makedirs(depth_path, exist_ok=True)

img_path_list = os.listdir(img_main_path)
file_length = len(img_path_list)
visibility_list = np.linspace(50, 1500, file_length)
csv_file_path = Path(main_path + "/visibility_data.csv")
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Image_Name", "Visibility"])
    # ----------------- 加入進度條 ----------------- #
    for index, img_file in tqdm(enumerate(img_path_list), total=file_length, desc="處理圖像文件"):
        img_file_ = os.path.join(img_main_path, img_file)
        main_vis = visibility_list[index]
        img_sub_file_list = os.listdir(img_file_)
        hazy_save_path = os.path.join(str(hazy_path)+'/' , img_file)
        depth_save_path = os.path.join(str(depth_path)+ '/' , img_file)
        os.makedirs(hazy_save_path, exist_ok=True)
        os.makedirs(depth_save_path, exist_ok=True)

        for img_name in tqdm(img_sub_file_list, desc=f"處理 {img_file}", leave=False):
            img_path = os.path.join(img_main_path, img_file, img_name)
            random_number = np.random.uniform(-1.5, 1.5)
            visibility = random_number + main_vis
            generate_fog_process(visibility, img_path, hazy_save_path, depth_save_path)
            writer.writerow([img_path, visibility])

