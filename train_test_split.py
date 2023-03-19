from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
BASE_PATH = 'E:\\video_classifier\data2'
VIDEOS_PATH = os.path.join(BASE_PATH, '**','*.mpg')
video_paths = tf.io.gfile.glob(VIDEOS_PATH)
print(video_paths)
train_names,test_names=train_test_split(video_paths,test_size=0.15)

with open('E:\\video_classifier\data2\\testlist.txt','w') as f:
    for i in test_names:
        f.write(i)
        f.write('\n')
with open('E:\\video_classifier\data2\\trainlist.txt','w') as f:
    for i in train_names:
        f.write(i)
        f.write('\n')