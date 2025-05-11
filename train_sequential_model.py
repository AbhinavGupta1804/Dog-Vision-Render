import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tf_keras.preprocessing.image import load_img, img_to_array
from tf_keras.utils import to_categorical
from tf_keras.src.callbacks import ModelCheckpoint
import numpy as np
from tqdm import tqdm
import tf_keras

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def load_dataset(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    breeds = sorted(df['breed'].unique())
    label_to_idx = {breed: idx for idx, breed in enumerate(breeds)}

    image_paths = [os.path.join(image_dir, f"{img_id}.jpg") for img_id in df['id']]
    labels = [label_to_idx[breed] for breed in df['breed']]
    labels = to_categorical(labels, num_classes=len(breeds))

    return image_paths, labels, breeds


def create_dataset(paths, labels, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    if training:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset


def build_model(num_classes):
    base_model = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5", trainable=False)
    model = tf_keras.Sequential([
        tf_keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        base_model,
        tf_keras.layers.Dense(256, activation='relu'),
        tf_keras.layers.Dropout(0.3),
        tf_keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_model(train_ds, val_ds, num_classes):
    model = build_model(num_classes)
    model.compile(optimizer=tf_keras.optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    checkpoint_path = os.path.join("model", "sequentialapi_model.h5")

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy',
                                 save_best_only=True, mode='max', verbose=1)
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True)

    history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[checkpoint])
    return model


def main():
    csv_path = r'C:\Users\abhi1\Desktop\CNN\labels.csv'
    image_dir = r'C:\Users\abhi1\Desktop\CNN\train'

    image_paths, labels, breeds = load_dataset(csv_path, image_dir)
    train_p, val_p, train_l, val_l = train_test_split(image_paths, labels, test_size=0.2, stratify=labels)

    train_ds = create_dataset(train_p, train_l, training=True)
    val_ds = create_dataset(val_p, val_l, training=False)

    model = train_model(train_ds, val_ds, num_classes=len(breeds))

    label_map = {i: b for i, b in enumerate(breeds)}
    np.save('label_map.npy', label_map)

if __name__ == '__main__':
    main()
