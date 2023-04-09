import tensorflow as tf
import os,cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
#from tensorflow_docs.vis import embed
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt

BASE_PATH = 'E:\\video_classifier\data2'
VIDEOS_PATH = os.path.join(BASE_PATH, '**','*.mpg')
SEQUENCE_LENGTH = 30
NUM_FEATURES = 2048
EPOCHS=50
classnames=os.listdir(BASE_PATH)

encoder = LabelBinarizer()
encoder.fit(classnames)

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

#define transformer models
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

def get_compiled_model():
    sequence_length=SEQUENCE_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = 10
    num_heads = 3
    classes = len(classnames)

    inputs = keras.Input(shape=(SEQUENCE_LENGTH,NUM_FEATURES))
    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.15)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    return model


def run_experiment():
    filepath = "./transformers_log"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    model = get_compiled_model()
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint],validation_data=valid_dataset
    )


    model.load_weights(filepath)
    _, accuracy = model.evaluate(valid_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return model

trained_model = run_experiment()





def predict_action(path):
    class_vocab = classnames
    model = get_compiled_model()
    filepath = "./transformers_log"
    model.load_weights(filepath)


    frames = np.load(path)
    frames=np.expand_dims(frames,0)
    probabilities = trained_model.predict(frames)[0]


    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames

test_frames = predict_action("E:\\video_classifier\data2\swing\\v_swing_01_01.npy")
