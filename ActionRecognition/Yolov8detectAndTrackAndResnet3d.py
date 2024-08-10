import cv2,torch
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from torchvision import transforms
from torchvision.models.video import r3d_18
import torch.nn as nn
import torchvision.io as io
def crop_and_pad(frame, box, margin_percent):
    """Crop box with margin and take square crop from frame."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1

    # Add margin
    margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
    x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)

    # Take square crop from frame
    size = max(y2 - y1, x2 - x1)
    center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
    half_size = size // 2
    square_crop = frame[
                  max(0, center_y - half_size):min(frame.shape[0], center_y + half_size),
                  max(0, center_x - half_size):min(frame.shape[1], center_x + half_size),
                  ]

    return cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_LINEAR)


def process_video(source: str, output_path: str, weights: str = "yolov8n.pt", device: str = "",
                  crop_margin_percentage: int = 10):
    # Initialize device and model
    device = select_device(device)
    yolo_model = YOLO(weights).to(device)

    # Initialize video capture
    cap = cv2.VideoCapture(source)

    # Get video properties
    frame_width = 224
    frame_height = 224
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize VideoWriter to save cropped video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize track history
    track_history = defaultdict(list)
    frame_counter = 0
    skip_frame = 2

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_counter += 1

        # Run YOLO detection and tracking
        results = yolo_model.track(frame, persist=True, classes=[0])  # Track only person class

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            # Save cropped video sequences
            if frame_counter % skip_frame == 0:
                for box, track_id in zip(boxes, track_ids):
                    crop = crop_and_pad(frame, box, crop_margin_percentage)
                    track_history[track_id].append(crop)

                    # Write the cropped frame to the video file
                    out.write(crop)

    # Release the resources
    cap.release()
    out.release()

def predict_new_video(model, video_path, transform, frames_per_clip=8):
    model.eval()
    video, _, info = io.read_video(video_path, pts_unit='sec')
    num_frames = video.shape[0]

    if num_frames >= frames_per_clip:
        start_idx = torch.randint(0, num_frames - frames_per_clip + 1, (1,)).item()
        video_clip = video[start_idx:start_idx + frames_per_clip]
    else:
        video_clip = torch.zeros((frames_per_clip, *video.shape[1:]))
        video_clip[:num_frames] = video
    print(video_clip.shape)
    if transform:
        video_clip = transform(video_clip.permute(0, 3, 1, 2).float())  # 改變維度順序
        video_clip = video_clip.permute(1, 0, 2, 3)  # 恢復維度順序，將 (T, C, H, W) 轉換為 (C, T, H, W)

    video_clip = video_clip.unsqueeze(0).to(device)  # 增加 batch 維度
    outputs = model(video_clip)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()
transform = transforms.Compose([
    transforms.Resize((224, 224)),

])

class_names=['walking', 'horse_riding', 'tennis', 'basketball']
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = r3d_18(pretrained=False)
num_classes = 4  # 替換為你的類別數
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('best_r3d18_model.pth'))
model.to(device)
process_video('/home/punzeonlung/fog_paper/testvideo/walking/v_walk_dog_01_01.avi',output_path='./testvideo.avi',weights='./yolov8m.pt',device=device)

predicted_class_idx = predict_new_video(model,'./testvideo.avi' , transform)
print(predicted_class_idx)
predicted_class_name = class_names[predicted_class_idx]

print(f'The predicted class for the new video is: {predicted_class_name}')