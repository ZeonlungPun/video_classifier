from __future__ import print_function
import os
import sys
import glob
import argparse
from multiprocessing import Pool, current_process
import cv2
import numpy as np


def run_optical_flow(vid_item):
    vid_path, vid_id = vid_item
    vid_name = os.path.splitext(os.path.basename(vid_path))[0]
    class_name=vid_path.split('/')[2]
    out_full_path = os.path.join(out_path,class_name, vid_name)
    os.makedirs(out_full_path, exist_ok=True)

    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // 20)  # Calculate step size to get 20 frames

    ret, prev_frame = cap.read()
    if not ret:
        print(f"Failed to read video {vid_path}")
        return False

    prev_frame = cv2.resize(prev_frame, new_size)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_idx = 0
    sampled_frames = 0

    while cap.isOpened() and sampled_frames < 20:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            frame = cv2.resize(frame, new_size)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            flow_x = os.path.join(out_full_path, f"flow_x_{sampled_frames:05d}.npy")
            flow_y = os.path.join(out_full_path, f"flow_y_{sampled_frames:05d}.npy")
            np.save(flow_x, flow[..., 0])
            np.save(flow_y, flow[..., 1])

            prev_gray = gray
            sampled_frames += 1

        frame_idx += 1

    cap.release()
    print(f"{vid_id} {vid_name} done")
    sys.stdout.flush()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument("--src_dir", type=str, default='./testvideo', help='path to the video data')
    parser.add_argument("--out_dir", type=str, default='./flow', help='path to store frames and optical flow')
    parser.add_argument("--new_width", type=int, default=250, help='resize image width')
    parser.add_argument("--new_height", type=int, default=250, help='resize image height')
    parser.add_argument("--num_worker", type=int, default=8, help='number of workers')
    parser.add_argument("--ext", type=str, default='avi', choices=['avi', 'mp4'], help='video file extensions')

    args = parser.parse_args()

    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    ext = args.ext
    new_size = (args.new_width, args.new_height)

    if not os.path.isdir(out_path):
        print(f"Creating folder: {out_path}")
        os.makedirs(out_path)

    vid_list = glob.glob(os.path.join(src_path, '*', f'*.{ext}'))
    print(f"Found {len(vid_list)} videos.")

    pool = Pool(num_worker)
    pool.map(run_optical_flow, zip(vid_list, range(len(vid_list))))
