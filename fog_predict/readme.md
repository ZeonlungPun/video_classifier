# Introduction

This is the official implementation of the paper 'Enhanced Visibility Prediction through Dynamic Video Analysis: Integrating Multi-stream Networks for Airport Surveillance' which foucus on leveraging video understaning technique to the domain of Visibility Prediction.

# Data

The data sources from part of question E from the 17th Chinese Graduate Mathematical Modeling Contest. You can just download here: https://www.shumo.com/wiki/doku.php?id=%E7%AC%AC%E5%8D%81%E4%B8%83%E5%B1%8A_2020_%E5%85%A8%E5%9B%BD%E7%A0%94%E7%A9%B6%E7%94%9F%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%E7%AB%9E%E8%B5%9B_npmcm_%E8%AF%95%E9%A2%98

# Folder Structure
You need to let the folder in this Structure Tree:
video samples:
```
video_class
├── 0
├── 1
└── 2
```
SIFT samples:
```
SIFT
├── 0
├── 1
└── 2
```
optical flow samples:
```
flow
├── 0
├── 1
└── 2
```


# Important Reference Paper

Palvanov, A., Cho, Y.I.:   Visnet: Deep convolutional neural networks for forecast-ing atmospheric visibility. Sensors 19(6), 1343 (2019)

Simonyan, K., Zisserman, A.:  Two-stream convolutional networks for action recognition in videos. Advances in neural information processing systems 27 (2014)

Liu, Z., Chen, Y., Gu, X., Yeoh, J.K., Zhang, Q.:  Visibility classification and influencing-factors analysis of airport: A deep learning approach. Atmospheric Environment 278, 119085 (2022)

# Model

We just implement our Three-stream Network in VisNet.py. 

two stream implementation (ResNet-50 as backbone): two_stream.py 

single stream implementation: siftnet.py , flownet.py

# Generate SIFT descriptor and optical flow data 

You need to get the optical flow data in generate.py, and get the SIFT descriptor tensor data in generate_SIFT.py.

```
parser.add_argument("--src_dir", type=str, default='./video_class', help='path to the video data')
parser.add_argument("--out_dir", type=str, default='./SIFT', help='path to store frames and SIFT descriptors')
```

# Environment

Ubuntu 22.04, NVIDIA GeForce RTX 3070-Ti  
```
albumentations                 1.3.1

cvzone                         1.6.1

Cython                         3.0.7

matplotlib                     3.7.2

numpy                          1.24.4

opencv-contrib-python-headless 4.10.0.84

opencv-python                  4.8.0.74

opencv-python-headless         4.8.1.78


pandas                         2.0.3

pycocotools                    2.0

scikit-image                   0.21.0

scikit-learn                   1.3.0

scipy                          1.10.1

seaborn                        0.12.2

torch                          2.2.2

torchaudio                     2.0.2

torchinfo                      1.8.0

torchmetrics                   1.3.1

torchvision                    0.17.2

```


