import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# PARAMETERS
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_EPOCHS   = 10
CSV_PATH     = "labels.csv"
IMAGE_DIR    = "train"
CHECKPOINT   = "best_model_tf"

df = pd.read_csv(CSV_PATH)
breeds = sorted(df["breed"].unique())
label_to_idx = {b:i   for i,b in enumerate(breeds)}  # i -> index , b -> breedname 
df["label"] = df["breed"].map(label_to_idx)          #replacing each name with its corresponding index

# SPLIT
paths = [os.path.join(IMAGE_DIR, f"{i}.jpg") for i in df["id"]]
labels = df["label"].values                                          #.values to convert into numpy , for traintestsplit
train_p, val_p, train_l, val_l = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=42   #stratify for balenced split
)

# DATASET PIPELINE
def preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)/255.0
    return img, label

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_p, train_l))
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)                                 #num_parallel_calls=tf.data.AUTOTUNE makes map() apply preprocessing to multiple elements in parallel for faster data loading, 
    .shuffle(1_000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)                                                           ## Prefetch to overlap data preprocessing and model training for faster execution
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_p, val_l))
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

FEATURE_EXTRACTOR_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
#this layer is primarily used to integrate a pre-trained model (from TensorFlow Hub) into your model as a feature extractor.
# i have made a custom layer so I can save and load the model easily (serialization) and reuse it later without errors

class FeatureExtractorLayer(keras.layers.Layer):
    #You're creating a custom layer by extending keras.layers.Layer
    def __init__(self, feature_extractor_url=None, trainable=False, **kwargs): #kwargs -> keywords & arguments
        super(FeatureExtractorLayer, self).__init__(**kwargs)                  #It calls the parent class (keras.layers.Layer) so the base functionality of a Keras Layer is initialized.
        self.feature_extractor_url = feature_extractor_url
        self.trainable_setting = trainable
        
        # Create feature extractor if URL is provided
        if feature_extractor_url:
            self.feature_extractor = hub.KerasLayer(
                feature_extractor_url, 
                trainable=trainable
            )
        else:
            self.feature_extractor = None

    def build(self, input_shape):
        #If I haven’t created the feature extractor yet, but the user has given me a URL to load it from — then go ahead and create it now
        if self.feature_extractor is None and self.feature_extractor_url:
            self.feature_extractor = hub.KerasLayer(
                self.feature_extractor_url,
                trainable=self.trainable_setting
            )
        super().build(input_shape)         #calling build method of parent class
        
    #forward pass  inputs = [224x224x3 image] → [1280-dim feature vector]
    def call(self, inputs):
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not initialized")
        return self.feature_extractor(inputs)                        #Applies the feature extractor model (like MobileNet) to the input images and returns feature vectors.
    
    #Used when saving/loading the model. It returns a dictionary with layer settings so that it can be rebuilt later.
    def get_config(self):
        config = super().get_config()
        config.update({
            'feature_extractor_url': self.feature_extractor_url,
            'trainable': self.trainable_setting
        })
        return config
    
    #This recreates the layer using the config returned by get_config() — required for saving/loading models.
    @classmethod
    def from_config(cls, config):
        # Extract the necessary configuration
        url = config.pop('feature_extractor_url', None)
        trainable = config.pop('trainable', False)
        return cls(feature_extractor_url=url, trainable=trainable, **config)

# Create the model
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = FeatureExtractorLayer(feature_extractor_url=FEATURE_EXTRACTOR_URL, trainable=False)(inputs)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(len(breeds), activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# CALLBACKS
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_model_tf.weights.h5",
    save_best_only=True,
    save_weights_only=True
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)

# TRAIN
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# SAVE FINAL
save_dir = "./model"
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, "functionalapi_model.h5"))


# EVALUATION
from sklearn.metrics import accuracy_score
predictions = model.predict(val_ds, verbose=1)
y_pred = np.argmax(predictions, axis=1)     #picks max prob index 
y_true = []
for _, labels in val_ds:
    y_true.extend(labels.numpy())           #.extend() adds each individual element from the array to the list.

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

