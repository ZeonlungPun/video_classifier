import argparse
import time
from collections import defaultdict
from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from torchvision import transforms

class TorchVisionVideoClassifier:
    """Classifies videos using pretrained TorchVision models; see https://pytorch.org/vision/stable/."""

    from torchvision.models.video import (
        MViT_V1_B_Weights,
        MViT_V2_S_Weights,
        R3D_18_Weights,
        S3D_Weights,
        Swin3D_B_Weights,
        Swin3D_T_Weights,
        mvit_v1_b,
        mvit_v2_s,
        r3d_18,
        s3d,
        swin3d_b,
        swin3d_t,
    )

    model_name_to_model_and_weights = {
        "s3d": (s3d, S3D_Weights.DEFAULT),
        "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
        "swin3d_t": (swin3d_t, Swin3D_T_Weights.DEFAULT),
        "swin3d_b": (swin3d_b, Swin3D_B_Weights.DEFAULT),
        "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights.DEFAULT),
        "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights.DEFAULT),
    }

    def __init__(self, model_name: str, device: str or torch.device = ""):
        """
        Initialize the VideoClassifier with the specified model name and device.

        Args:
            model_name (str): The name of the model to use.
            device (str or torch.device, optional): The device to run the model on. Defaults to "".

        Raises:
            ValueError: If an invalid model name is provided.
        """
        if model_name not in self.model_name_to_model_and_weights:
            raise ValueError(f"Invalid model name '{model_name}'. Available models: {self.available_model_names()}")
        model, self.weights = self.model_name_to_model_and_weights[model_name]
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model(weights=self.weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)
        self.model.load_state_dict(torch.load("./best_r3d18_model.pth"))
        self.model.to(self.device).eval()


    @staticmethod
    def available_model_names() -> List[str]:
        """
        Get the list of available model names.

        Returns:
            list: List of available model names.
        """
        return list(TorchVisionVideoClassifier.model_name_to_model_and_weights.keys())

    def preprocess_crops_for_video_cls(self, crops: List[np.ndarray], input_size: list = None) -> torch.Tensor:
        """
        Preprocess a list of crops for video classification.

        Args:
            crops (List[np.ndarray]): List of crops to preprocess. Each crop should have dimensions (H, W, C)
            input_size (tuple, optional): The target input size for the model. Defaults to (224, 224).

        Returns:
            torch.Tensor: Preprocessed crops as a tensor with dimensions (1, T, C, H, W).
        """
        if input_size is None:
            input_size = [224, 224]
        from torchvision.transforms import v2

        # transform = v2.Compose(
        #     [
        #         #v2.ToDtype(torch.float32, scale=True),
        #         v2.Resize(input_size, antialias=True),
        #         #v2.Normalize(mean=self.weights.transforms().mean, std=self.weights.transforms().std),
        #     ]
        # )
        transform = transforms.Compose([
            transforms.Resize((224, 224)),

        ])

        processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]
        return torch.stack(processed_crops).unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)

    def __call__(self, sequences: torch.Tensor):
        """
        Perform inference on the given sequences.

        Args:
            sequences (torch.Tensor): The input sequences for the model. The expected input dimensions are
                                      (B, T, C, H, W) for batched video frames or (T, C, H, W) for single video frames.

        Returns:
            torch.Tensor: The model's output.
        """
        with torch.inference_mode():
            return self.model(sequences)

    def postprocess(self, outputs: torch.Tensor,labels):
        """
        Postprocess the model's batch output.

        Args:
            outputs (torch.Tensor): The model's output.

        Returns:
            List[str]: The predicted labels.
            List[float]: The predicted confidences.
        """

        pred_labels = []
        pred_confs = []
        for output in outputs:
            pred_class = output.argmax(0).item()
            #pred_label = self.weights.meta["categories"][pred_class]
            pred_label = labels[pred_class]
            pred_labels.append(pred_label)
            pred_conf = output.softmax(0)[pred_class].item()
            pred_confs.append(pred_conf)

        return pred_labels, pred_confs





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
        max(0, center_y - half_size) : min(frame.shape[0], center_y + half_size),
        max(0, center_x - half_size) : min(frame.shape[1], center_x + half_size),
    ]

    return cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_LINEAR)


def run(
    weights: str = "./yolov8m.pt",
    device: str = "",
    source: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_path: Optional[str] = None,
    crop_margin_percentage: int = 10,
    num_video_sequence_samples: int = 8,
    skip_frame: int = 2,
    video_cls_overlap_ratio: float = 0.25,
    fp16: bool = False,
    video_classifier_model: str = "microsoft/xclip-base-patch32",
    labels: List[str] = ['walking', 'horse_riding', 'tennis', 'basketball'],
) -> None:
    """
    Run action recognition on a video source using YOLO for object detection and a video classifier.

    Args:
        weights (str): Path to the YOLO model weights. Defaults to "yolov8n.pt".
        device (str): Device to run the model on. Use 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, or 'cpu'. Defaults to auto-detection.
        source (str): Path to mp4 video file or YouTube URL. Defaults to a sample YouTube video.
        output_path (Optional[str], optional): Path to save the output video. Defaults to None.
        crop_margin_percentage (int, optional): Percentage of margin to add around detected objects. Defaults to 10.
        num_video_sequence_samples (int, optional): Number of video frames to use for classification. Defaults to 8.
        skip_frame (int, optional): Number of frames to skip between detections. Defaults to 4.
        video_cls_overlap_ratio (float, optional): Overlap ratio between video sequences. Defaults to 0.25.
        fp16 (bool, optional): Whether to use half-precision floating point. Defaults to False.
        video_classifier_model (str, optional): Name or path of the video classifier model. Defaults to "microsoft/xclip-base-patch32".
        labels (List[str], optional): List of labels for zero-shot classification. Defaults to predefined list.

    Returns:
        None</edit>
    """

    # Initialize models and device
    # device = select_device(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    yolo_model = YOLO(weights).to(device)

    video_classifier = TorchVisionVideoClassifier(video_classifier_model, device=device)

    # Initialize video capture
    cap = cv2.VideoCapture(source)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize VideoWriter
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize track history
    track_history = defaultdict(list)
    frame_counter = 0

    track_ids_to_infer = []
    crops_to_infer = []
    pred_labels = []
    pred_confs = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_counter += 1

        # Run YOLO tracking
        results = yolo_model.track(frame, persist=True, classes=[0])  # Track only person class

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            # Visualize prediction
            annotator = Annotator(frame, line_width=3, font_size=10, pil=False)

            if frame_counter % skip_frame == 0:
                crops_to_infer = []
                track_ids_to_infer = []

            for box, track_id in zip(boxes, track_ids):
                if frame_counter % skip_frame == 0:
                    crop = crop_and_pad(frame, box, crop_margin_percentage)
                    track_history[track_id].append(crop)

                if len(track_history[track_id]) > num_video_sequence_samples:
                    track_history[track_id].pop(0)

                if len(track_history[track_id]) == num_video_sequence_samples and frame_counter % skip_frame == 0:
                    start_time = time.time()
                    crops = video_classifier.preprocess_crops_for_video_cls(track_history[track_id])
                    end_time = time.time()
                    preprocess_time = end_time - start_time
                    print(f"video cls preprocess time: {preprocess_time:.4f} seconds")
                    crops_to_infer.append(crops)
                    track_ids_to_infer.append(track_id)

            if crops_to_infer and (
                not pred_labels
                or frame_counter % int(num_video_sequence_samples * skip_frame * (1 - video_cls_overlap_ratio)) == 0
            ):
                crops_batch = torch.cat(crops_to_infer, dim=0).float()

                start_inference_time = time.time()
                output_batch = video_classifier(crops_batch)
                end_inference_time = time.time()
                inference_time = end_inference_time - start_inference_time
                print(f"video cls inference time: {inference_time:.4f} seconds")

                pred_labels, pred_confs = video_classifier.postprocess(output_batch,labels)
                print(pred_labels, pred_confs)

            if track_ids_to_infer and crops_to_infer:
                for box, track_id, pred_label, pred_conf in zip(boxes, track_ids_to_infer, pred_labels, pred_confs):
                    top2_preds = zip([pred_label], [pred_conf])
                    label_text = " | ".join([f"{label} ({conf:.2f})" for label, conf in top2_preds])
                    annotator.box_label(box, label_text, color=(0, 0, 255))

        # Write the annotated frame to the output video
        if output_path is not None:
            out.write(frame)



    cap.release()
    if output_path is not None:
        out.release()
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="ultralytics detector model path")
    parser.add_argument("--device", default="cuda:0", help='cuda device, i.e. 0 or 0,1,2,3 or cpu/mps, "" for auto-detection')
    parser.add_argument(
        "--source",
        type=str,
        default="/home/punzeonlung/fog_paper/testvideo/horse_riding/v_riding_01_01.avi",
        help="video file path or youtube URL",
    )
    parser.add_argument("--output-path", type=str, default="output_video.mp4", help="output video file path")
    parser.add_argument(
        "--crop-margin-percentage", type=int, default=10, help="percentage of margin to add around detected objects"
    )
    parser.add_argument(
        "--num-video-sequence-samples", type=int, default=8, help="number of video frames to use for classification"
    )
    parser.add_argument("--skip-frame", type=int, default=2, help="number of frames to skip between detections")
    parser.add_argument(
        "--video-cls-overlap-ratio", type=float, default=0.25, help="overlap ratio between video sequences"
    )
    parser.add_argument("--fp16", action="store_true", help="use FP16 for inference")
    parser.add_argument(
        "--video-classifier-model", type=str, default="r3d_18", help="video classifier model name"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        default=['walking', 'horse_riding', 'tennis', 'basketball'],
        help="labels for zero-shot video classification",
    )
    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
