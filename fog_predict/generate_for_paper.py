import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_center(img, crop_width, crop_height):
    h, w = img.shape[:2]
    startx = w // 2 - (crop_width // 2)
    starty = h // 2 - (crop_height // 2)
    return img[starty:starty + crop_height, startx:startx + crop_width]

def draw_optical_flow(img, flowx, flowy, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flowx[y, x], flowy[y, x]

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis

def flow_to_color(flowx, flowy):
    h, w = flowx.shape
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[..., 0] = flowx
    flow[..., 1] = flowy

    hsv = np.zeros((h, w, 3), dtype=np.float32)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 1
    hsv[..., 2] = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

# 讀取灰度圖像
img1 = cv2.imread('/home/punzeonlung/fog_paper/abnormal1.jpg', cv2.IMREAD_GRAYSCALE)
img2= cv2.imread('/home/punzeonlung/fog_paper/abnormal2.jpg', cv2.IMREAD_GRAYSCALE)
# # 截取中間部分 w=600, h=200
crop_width, crop_height = 900, 400
img1_cropped = crop_center(img1, crop_width, crop_height)
img2_cropped = crop_center(img2, crop_width, crop_height)
cv2.imwrite('abnormal1_.png',img1_cropped)
cv2.imwrite('abnormal2_.png',img2_cropped)



# # 加載光流數據
#flowx = np.load('/home/punzeonlung/fog_paper/flow/0/video1/flow_x_00000.npy')
#flowy = np.load('/home/punzeonlung/fog_paper/flow/0/video1/flow_y_00000.npy')

# # 確保 flowx 和 flowy 是數組
# if not isinstance(flowx, np.ndarray) or not isinstance(flowy, np.ndarray):
#     raise TypeError("flowx and flowy should be numpy arrays")
#
# # 截取光流數據的中間部分
#flowx_cropped = crop_center(flowx, crop_width, crop_height)
#flowy_cropped = crop_center(flowy, crop_width, crop_height)
#
# # 可視化光流矢量場
# flow_vis = draw_optical_flow(img1_cropped, flowx_cropped, flowy_cropped)
#
# # 將光流轉換為顏色編碼圖像
# flow_color = flow_to_color(flowx_cropped, flowy_cropped)
#
# # 顯示結果
# cv2.imshow('Optical Flow Vector', flow_vis)
# cv2.imshow('Optical Flow Color', flow_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 或者使用matplotlib顯示
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.title('Optical Flow Vector')
# plt.imshow(flow_vis[..., ::-1])  # cv2默認是BGR, 這裡轉換為RGB顯示
#
# plt.subplot(1, 2, 2)
# plt.title('Optical Flow Color')
# plt.imshow(flow_color)
#
# plt.show()
# import cv2
#
#
# # 讀取灰度圖像
# img = cv2.imread('./flow1raw.png')
# img = cv2.fastNlMeansDenoisingColored(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
# img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# # 應用 CLAHE 對 Y 通道（亮度）進行增強
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
# img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
# # 將圖像轉換回 BGR 色彩空間
# img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
# # 創建SIFT對象
# sift = cv2.SIFT_create()
#
# # 檢測關鍵點和計算描述符
# kp1, des1 = sift.detectAndCompute(img, None)
# print(des1.shape )
# # 在圖像上繪制關鍵點
# img_with_kp = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# # 顯示結果
# cv2.imwrite('sift_vis.png',img_with_kp)