This is the official implementation of the paper 'Enhanced Visibility Prediction through Dynamic Video Analysis: Integrating Multi-stream Networks for Airport Surveillance' which foucus on leveraging video understaning technique to the domain of Visibility Prediction.

# Data

The data sources from part of question E from the 17th Chinese Graduate Mathematical Modeling Contest. You can just download here: https://www.shumo.com/wiki/doku.php?id=%E7%AC%AC%E5%8D%81%E4%B8%83%E5%B1%8A_2020_%E5%85%A8%E5%9B%BD%E7%A0%94%E7%A9%B6%E7%94%9F%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%E7%AB%9E%E8%B5%9B_npmcm_%E8%AF%95%E9%A2%98

# Important Reference Paper

Palvanov, A., Cho, Y.I.: Visnet: Deep convolutional neural networks for forecast-ing atmospheric visibility. Sensors 19(6), 1343 (2019)

Simonyan, K., Zisserman, A.: Two-stream convolutional networks for action recognition in videos. Advances in neural information processing systems 27 (2014)

Liu, Z., Chen, Y., Gu, X., Yeoh, J.K., Zhang, Q.: Visibility classification and influencing-factors analysis of airport: A deep learning approach. Atmospheric
Environment 278, 119085 (2022)

# Model

We just implement our Three-stream Network in VisNet.py. You need to get the optical flow data in generate.py, and get the SIFT descriptor tensor data in generate_SIFT.py.

single stream implementation: siftnet.py , flownet.py

two stream implementation: two_stream.py 

# Environment

Ubuntu 22.04, NVIDIA GeForce RTX 3070-Ti  
"""
absl-py                        2.0.0
aiosignal                      1.3.1
albumentations                 1.3.1
appdirs                        1.4.4
asttokens                      2.4.1
attrs                          23.2.0
cachetools                     5.3.2
certifi                        2023.7.22
charset-normalizer             3.2.0
click                          8.1.7
clip                           1.0
cmake                          3.27.1
coloredlogs                    15.0.1
commonmark                     0.9.1
contourpy                      1.1.0
cvzone                         1.6.1
cycler                         0.11.0
Cython                         3.0.7
cython-bbox                    0.1.5
dataclasses-json               0.6.4
decorator                      5.1.1
docker-pycreds                 0.4.0
easydict                       1.12
et-xmlfile                     1.1.0
executing                      2.0.1
ExifRead                       3.0.0
faiss-gpu                      1.7.2
faster-coco-eval               1.4.3
filelock                       3.12.2
filterpy                       1.4.5
fire                           0.6.0
flatbuffers                    23.5.26
fonttools                      4.42.0
frozenlist                     1.4.1
fsspec                         2024.2.0
ftfy                           6.1.3
gdown                          4.7.1
gitdb                          4.0.11
GitPython                      3.1.40
google-auth                    2.23.3
google-auth-oauthlib           1.0.0
grpcio                         1.59.0
h5py                           3.11.0
hub-sdk                        0.0.2
huggingface-hub                0.21.3
humanfriendly                  10.0
idna                           3.4
imageio                        2.31.1
imgviz                         1.7.5
importlib-metadata             6.8.0
importlib-resources            6.0.0
imutils                        0.5.4
ipdb                           0.13.13
ipython                        8.12.3
Jinja2                         3.1.2
joblib                         1.3.2
jsonschema                     4.21.1
jsonschema-specifications      2023.12.1
kiwisolver                     1.4.4
lazy_loader                    0.3
lightgbm                       4.3.0
lightning-utilities            0.10.1
lit                            16.0.6
lxml                           4.9.3
Markdown                       3.5
MarkupSafe                     2.1.3
marshmallow                    3.21.1
matplotlib                     3.7.2
matplotlib-inline              0.1.6
motmetrics                     1.4.0
mpmath                         1.3.0
msgpack                        1.0.8
munch                          4.0.0
mypy-extensions                1.0.0
natsort                        8.4.0
networkx                       3.1
norfair                        2.2.0
numpy                          1.24.4
nvidia-cublas-cu11             11.10.3.66
nvidia-cublas-cu12             12.1.3.1
nvidia-cuda-cupti-cu11         11.7.101
nvidia-cuda-cupti-cu12         12.1.105
nvidia-cuda-nvrtc-cu11         11.7.99
nvidia-cuda-nvrtc-cu12         12.1.105
nvidia-cuda-runtime-cu11       11.7.99
nvidia-cuda-runtime-cu12       12.1.105
nvidia-cudnn-cu11              8.5.0.96
nvidia-cudnn-cu12              8.9.2.26
nvidia-cufft-cu11              10.9.0.58
nvidia-cufft-cu12              11.0.2.54
nvidia-curand-cu11             10.2.10.91
nvidia-curand-cu12             10.3.2.106
nvidia-cusolver-cu11           11.4.0.1
nvidia-cusolver-cu12           11.4.5.107
nvidia-cusparse-cu11           11.7.4.91
nvidia-cusparse-cu12           12.1.0.106
nvidia-nccl-cu11               2.14.3
nvidia-nccl-cu12               2.19.3
nvidia-nvjitlink-cu12          12.4.127
nvidia-nvtx-cu11               11.7.91
nvidia-nvtx-cu12               12.1.105
oauthlib                       3.2.2
onemetric                      0.1.2
onnxruntime                    1.15.1
opencv-contrib-python-headless 4.10.0.84
opencv-python                  4.8.0.74
opencv-python-headless         4.8.1.78
openpyxl                       3.1.2
packaging                      23.1
pandas                         2.0.3
parso                          0.8.3
patsy                          0.5.6
pexpect                        4.9.0
pickleshare                    0.7.5
Pillow                         9.5.0
pip                            23.2.1
pkgutil_resolve_name           1.3.10
plotly                         5.20.0
prefetch-generator             1.0.3
pretrainedmodels               0.7.4
prompt-toolkit                 3.0.43
protobuf                       4.24.0
psutil                         5.9.5
psycopg2                       2.9.3
ptyprocess                     0.7.0
pure-eval                      0.2.2
py-cpuinfo                     9.0.0
pyasn1                         0.5.0
pyasn1-modules                 0.3.0
pybboxes                       0.1.6
pycocotools                    2.0
Pygments                       2.17.2
pymssql                        2.2.5
pyodbc                         4.0.34
pyparsing                      3.0.9
referencing                    0.33.0
regex                          2023.12.25
requests                       2.31.0
requests-oauthlib              1.3.1
rich                           12.6.0
rpds-py                        0.18.0
rsa                            4.9
safetensors                    0.4.2
sahi                           0.11.16
scikit-image                   0.21.0
scikit-learn                   1.3.0
scipy                          1.10.1
seaborn                        0.12.2
sentry-sdk                     1.40.5
setproctitle                   1.3.3
setuptools                     68.0.0
shapely                        2.0.1
six                            1.16.0
smmap                          5.0.1
soupsieve                      2.4.1
stack-data                     0.6.3
statsmodels                    0.14.1
supervision                    0.1.0
sympy                          1.12
tabulate                       0.9.0
tenacity                       8.2.3
tensorboard                    2.14.0
tensorboard-data-server        0.7.2
tensorboardX                   2.6.2.2
termcolor                      2.3.0
terminaltables                 3.1.10
thop                           0.1.1.post2209072238
threadpoolctl                  3.2.0
tifffile                       2023.7.10
timm                           0.9.16
tokenizers                     0.19.1
tomli                          2.0.1
torch                          2.2.2
torchaudio                     2.0.2
torchinfo                      1.8.0
torchmetrics                   1.3.1
torchvision                    0.17.2

"""


