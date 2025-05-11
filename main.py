from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os
import pandas as pd
import random
import io
import base64

app = Flask(__name__)

# --------------------- FeatureExtractor Layer --------------------- #
class FeatureExtractorLayer(tf.keras.layers.Layer):
    def __init__(self, feature_extractor_url=None, trainable=False, **kwargs):
        super(FeatureExtractorLayer, self).__init__(**kwargs)
        self.feature_extractor_url = feature_extractor_url
        self.trainable_setting = trainable

        if feature_extractor_url:
            self.feature_extractor = hub.KerasLayer(
                feature_extractor_url,
                trainable=trainable
            )
        else:
            self.feature_extractor = None

    def build(self, input_shape):
        if self.feature_extractor is None and self.feature_extractor_url:
            self.feature_extractor = hub.KerasLayer(
                self.feature_extractor_url,
                trainable=self.trainable_setting
            )
        super().build(input_shape)

    def call(self, inputs):
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not initialized")
        return self.feature_extractor(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'feature_extractor_url': self.feature_extractor_url,
            'trainable': self.trainable_setting
        })
        return config

    @classmethod
    def from_config(cls, config):
        url = config.pop('feature_extractor_url', None)
        trainable = config.pop('trainable', False)
        return cls(feature_extractor_url=url, trainable=trainable, **config)

# --------------------- Load Model --------------------- #
def load_model():
    custom_objects = {
        "FeatureExtractorLayer": FeatureExtractorLayer
    }
    model_path = os.path.join("model", "functionalapi_model.h5")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

# --------------------- Load Breeds --------------------- #
def load_breeds():
    df = pd.read_csv("labels.csv")
    breeds = sorted(df["breed"].unique())
    return breeds

# --------------------- Load labels.csv once --------------------- #
labels_df = pd.read_csv("labels.csv")
model = load_model()
breeds = load_breeds()

# --------------------- Preprocess Image --------------------- #
def preprocess_image(image):
    IMG_SIZE = 224
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# --------------------- Get Breed Image --------------------- #
def get_breed_image(predicted_breed):
    breed_images = labels_df[labels_df['breed'] == predicted_breed]
    if not breed_images.empty:
        image_id = random.choice(breed_images['id'].tolist())
        image_path = os.path.join("train", f"{image_id}.jpg")
        if os.path.exists(image_path):
            img = Image.open(image_path).convert('RGB')
            # Convert PIL image to base64 for HTML display
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
    return None

# --------------------- Get Sample Image from breed_examples --------------------- #
def get_sample_image():
    # Check if breed_examples directory exists
    sample_dir = "breed_examples"
    if not os.path.exists(sample_dir) or not os.path.isdir(sample_dir):
        return None, "Sample directory not found"
    
    # Get all image files from the directory
    valid_extensions = ['.jpg', '.jpeg', '.png']
    sample_files = [f for f in os.listdir(sample_dir) 
                   if os.path.isfile(os.path.join(sample_dir, f)) and 
                   any(f.lower().endswith(ext) for ext in valid_extensions)]
    
    if not sample_files:
        return None, "No sample images found"
    
    # Pick a random sample
    sample_file = random.choice(sample_files)
    sample_path = os.path.join(sample_dir, sample_file)
    
    try:
        # Open and convert to base64
        img = Image.open(sample_path).convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img, f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        return None, f"Error loading sample image: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_sample_image', methods=['GET'])
def sample_image():
    img, image_data_or_error = get_sample_image()
    
    if img is None:
        return jsonify({'error': image_data_or_error})
    
    try:
        # Process the sample image
        processed_image = preprocess_image(img)
        
        # Make prediction
        prediction = model.predict(processed_image)
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            breed_name = breeds[idx]
            confidence = float(prediction[0][idx] * 100)
            breed_image = get_breed_image(breed_name)
            
            results.append({
                'rank': i+1,
                'breed': breed_name,
                'confidence': f"{confidence:.2f}%",
                'sample_image': breed_image
            })
        
        return jsonify({
            'success': True,
            'image_data': image_data_or_error,
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({'error': f"Error processing sample image: {str(e)}"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Read and process the image
        img = Image.open(file.stream).convert('RGB')
        processed_image = preprocess_image(img)
        
        # Make prediction
        prediction = model.predict(processed_image)
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            breed_name = breeds[idx]
            confidence = float(prediction[0][idx] * 100)
            breed_image = get_breed_image(breed_name)
            
            results.append({
                'rank': i+1,
                'breed': breed_name,
                'confidence': f"{confidence:.2f}%",
                'sample_image': breed_image
            })
        
        # Convert uploaded image to base64 for display
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        uploaded_image = f"data:image/jpeg;base64,{img_str}"
        
        return jsonify({
            'success': True,
            'uploaded_image': uploaded_image,
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False)