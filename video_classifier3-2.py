import tensorflow as tf
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer

BASE_PATH = 'E:\\video_classifier\data2'
VIDEOS_PATH = os.path.join(BASE_PATH, '**','*.mpg')
SEQUENCE_LENGTH = 30
classnames=os.listdir(BASE_PATH)
encoder = LabelBinarizer()
encoder.fit(classnames)
#Step 2: Train the LSTM on video features

model = tf.keras.Sequential([
    tf.keras.layers.Masking(mask_value=0.),
    tf.keras.layers.LSTM(512, dropout=0.5, recurrent_dropout=0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(classnames), activation='softmax')
])
model.build(input_shape=(None,SEQUENCE_LENGTH, 2048))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

def make_generator(file_list):
    def generator():
        np.random.shuffle(file_list)
        for path in file_list:
            full_path = path.replace('.mpg', '.npy')
            label = os.path.basename(os.path.dirname(path))
            features = np.load(full_path)

            padded_sequence = np.zeros((SEQUENCE_LENGTH, 2048))
            padded_sequence[0:len(features)] = np.array(features)

            transformed_label = encoder.transform([label])
            yield padded_sequence, transformed_label[0]
    return generator

test_file = os.path.join('E:\\video_classifier', 'testlist.txt')
train_file = os.path.join('E:\\video_classifier', 'trainlist.txt')

with open(train_file) as f:
    test_list = [row.strip() for row in list(f)]

with open(test_file) as f:
    train_list = [row.strip() for row in list(f)]
    train_list = [row.split(' ')[0] for row in train_list]


train_dataset = tf.data.Dataset.from_generator(make_generator(train_list),
                 output_types=(tf.float32, tf.int16),
                 output_shapes=((SEQUENCE_LENGTH, 2048), (len(classnames))))
train_dataset = train_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)


valid_dataset = tf.data.Dataset.from_generator(make_generator(test_list),
                 output_types=(tf.float32, tf.int16),
                 output_shapes=((SEQUENCE_LENGTH, 2048), (len(classnames))))
valid_dataset = valid_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)


model.fit(train_dataset, epochs=17, validation_data=valid_dataset)