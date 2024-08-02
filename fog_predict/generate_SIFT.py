from __future__ import print_function
import os
import sys
import glob
import argparse
from multiprocessing import Pool
import cv2
import numpy as np

# 全局變量保存關鍵點
saved_keypoints = None

def run_SIFT(vid_item):
    vid_path, vid_id = vid_item
    vid_name = os.path.splitext(os.path.basename(vid_path))[0]
    class_name = vid_path.split('/')[2]
    out_full_path = os.path.join(out_path, class_name, vid_name)
    os.makedirs(out_full_path, exist_ok=True)

    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)  # Calculate step size to get num_frames frames

    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read video {vid_path}")
        return False

    # Get the middle part of the frame with size new_size
    height, width, _ = frame.shape
    new_height, new_width = new_size
    center_x, center_y = width // 2, height // 2
    crop_x1 = center_x - new_width // 2
    crop_y1 = center_y - new_height // 2
    crop_x2 = center_x + new_width // 2
    crop_y2 = center_y + new_height // 2
    frame_cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    # frame_cropped = cv2.fastNlMeansDenoisingColored(frame_cropped, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # img_yuv = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2YUV)
    # # 應用 CLAHE 對 Y 通道（亮度）進行增強
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    # img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    # # 將圖像轉換回 BGR 色彩空間
    # frame_cropped = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    frame_idx = 0
    sampled_frames = 0

    while cap.isOpened() and sampled_frames < num_frames:
        if frame_idx % step == 0:
            sift = cv2.SIFT_create()
            _, des = sift.compute(frame_cropped, saved_keypoints)
            dest_path = os.path.join(out_full_path, f"SIFT_{sampled_frames:05d}.npy")
            np.save(dest_path, des)
            sampled_frames += 1

        frame_idx += 1

    cap.release()
    print(f"{vid_id} {vid_name} done")
    sys.stdout.flush()
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract SIFT descriptors")
    parser.add_argument("--src_dir", type=str, default='./video_class', help='path to the video data')
    parser.add_argument("--out_dir", type=str, default='./SIFT', help='path to store frames and SIFT descriptors')
    parser.add_argument("--new_width", type=int, default=200, help='resize image width')
    parser.add_argument("--new_height", type=int, default=600, help='resize image height')
    parser.add_argument("--num_worker", type=int, default=8, help='number of workers')
    parser.add_argument("--num_frames", type=int, default=15, help='number of frames to sample per video')
    parser.add_argument("--ext", type=str, default='avi', choices=['avi', 'mp4'], help='video file extensions')

    args = parser.parse_args()

    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    ext = args.ext
    new_size = (args.new_width, args.new_height)
    num_frames = args.num_frames

    img = cv2.imread('/home/kingargroo/fog/temp/frame_35.jpg')
    height, width, _ = img.shape
    sift = cv2.SIFT_create()
    new_height, new_width = new_size
    print(new_size)
    center_x, center_y = width // 2, height // 2
    crop_x1 = center_x - new_width // 2
    crop_y1 = center_y - new_height // 2
    crop_x2 = center_x + new_width // 2
    crop_y2 = center_y + new_height // 2
    img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    img = cv2.fastNlMeansDenoisingColored(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # 應用 CLAHE 對 Y 通道（亮度）進行增強
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    # 將圖像轉換回 BGR 色彩空間
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    kp1, des1 = sift.detectAndCompute(img, None)
    saved_keypoints = kp1

    if not os.path.isdir(out_path):
        print(f"Creating folder: {out_path}")
        os.makedirs(out_path)

    vid_list = glob.glob(os.path.join(src_path, '*', f'*.{ext}'))
    print(f"Found {len(vid_list)} videos.")

    pool = Pool(num_worker)
    pool.map(run_SIFT, zip(vid_list, range(len(vid_list))))
