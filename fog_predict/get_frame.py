import pandas as pd
import numpy as np
import cv2,os,shutil,re

vis_data=pd.read_csv('./rvr_data.csv')
vis_data['LOCALDATE (BEIJING)']=pd.to_datetime(vis_data['LOCALDATE (BEIJING)'])
#vis_data=vis_data[ (vis_data['LOCALDATE (BEIJING)']>'3/13/2020 00:00:00 AM') & (vis_data['LOCALDATE (BEIJING)'] <'3/13/2020 04:00:00 AM') &(vis_data['LOCALDATE (BEIJING)'] != '3/13/2020 01:53:00 AM') ]
vis_data.index=vis_data['LOCALDATE (BEIJING)']
RVR_DATA=  vis_data['RVR DATA']



class_list=[]
for vis in np.array(RVR_DATA)[:600]:
    if vis>1000:
        class_list.append(0)
    elif vis<=1000 and vis>=500:
        class_list.append(1)
    elif vis <= 300:
        class_list.append(2)
    # else:
    #     class_list.append(3)
print("0:",np.sum(np.array(class_list)==0))
print("1:",np.sum(np.array(class_list)==1))
print("2:",np.sum(np.array(class_list)==2))
print("3:",np.sum(np.array(class_list)==3))





# # 設置影片路徑和輸出文件夾
# video_path = 'fog.mp4'
# output_dir = './temp5'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
#
# cap = cv2.VideoCapture(video_path)
#
# # 檢查影片是否成功讀取
# if not cap.isOpened():
#     print("Error: Could not open video.")
# else:
#     # 獲取影片訊息
#     frame_rate = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     duration = frame_count / frame_rate
#
#     print(f"Frame Rate: {frame_rate} FPS")
#     print(f"Frame Count: {frame_count}")
#     print(f"Duration: {duration} seconds")
#
#     # 計算目標時間點的幀索引
#     target_times = []
#     start_time = 1*3600+39*60+42
#     end_time = 3 * 3600 + 25 * 60 + 42
#     interval = 1  # 每15秒保存一幀
#
#     current_time = start_time
#     while current_time <= end_time:
#         target_times.append(current_time)
#         current_time += interval
#
#     # 保存幀
#     for target_time in target_times:
#         cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
#         ret, frame = cap.read()
#         if ret:
#             frame_name = os.path.join(output_dir, f"frame_{int(target_time+5955)}.jpg")
#             cv2.imwrite(frame_name, frame)
#         else:
#             print(f"Warning")

# # 設置時間間隔和邊界
# time_intervals = [
#     (0, '1:39:42', '3:25:42', 1),  # 每1秒保存一幀
#     #(1, '1:39:15', '7:45:27', 15)  # 每15秒保存一幀
# ]
#
#
# # 將時間轉換為秒數
# def time_to_seconds(time_str):
#     h, m, s = map(int, time_str.split(':'))
#     return h * 3600 + m * 60 + s
#
#
# # 讀取影片
# cap = cv2.VideoCapture(video_path)
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# if not cap.isOpened():
#     print("Error: Could not open video.")
# else:
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         current_time = frame_count / fps
#
#         # 檢查當前時間落在哪個區間
#         save_interval = None
#         for interval in time_intervals:
#             if time_to_seconds(interval[1]) <= current_time < time_to_seconds(interval[2]):
#                 save_interval = interval[3]
#                 break
#
#         if save_interval and frame_count % (fps * save_interval) == 0:
#             output_path = os.path.join(output_dir, f'frame_{int(current_time+5955)}.jpg')
#             cv2.imwrite(output_path, frame)
#
#         frame_count += 1
#
# cap.release()
# print("幀圖片已成功保存。")

# 設置來源文件夾和目標文件夾
# source_dir = './temp5'
# destination_dir = './frames'
#
# # 獲取來源文件夾內的所有圖片文件，並排序
# image_files = sorted([f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))])
# # 提取文件名中的數字並排序
# def extract_number(filename):
#     match = re.search(r'frame_(\d+)\.jpg', filename)
#     return int(match.group(1)) if match else -1
#
# image_files.sort(key=extract_number)
# # 每60張圖片保存到一個子文件夾
#
# group_size = 15
# for i in range(0, len(image_files), group_size):
#     # 創建子文件夾名稱
#     subfolder_name = f'video{i // group_size + 1+390}'
#     subfolder_path = os.path.join(destination_dir, subfolder_name)
#
#     # 創建子文件夾
#     os.makedirs(subfolder_path, exist_ok=True)
#
#     # 複製每組的圖片到子文件夾
#     for image_file in image_files[i:i + group_size]:
#         source_file = os.path.join(source_dir, image_file)
#         destination_file = os.path.join(subfolder_path, image_file)
#         shutil.copy2(source_file, destination_file)
#
# print("圖片已成功分組並保存到子文件夾。")

data1=pd.read_csv('./rvr_data.csv')
data2=pd.read_csv('./VIS_R06_12.csv')

data=pd.merge(left=data1,right=data2,on='LOCALDATE (BEIJING)',how='inner')