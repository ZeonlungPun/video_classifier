# Action Recognition 
This reprository contains some files that are needed to train your own Action Recognition Model (mainly vedio classifier such as r3d18,S3D and so on).

# Composition

An Action Recognition Model mainly contains object detection model (here is Yolov8), object tracking model (here are ByteTrack and BotSort), and a 
vedio classifier model (some pretrain models from torchvision.models.video module).

# Process
1, Train a specific object detection model

2, Use object detection model and object tracking model to extract ROI sequences

3， Train a vedio classifier model with extracted ROI sequences 

# Demo




