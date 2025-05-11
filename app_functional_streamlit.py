import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

# Add current directory to path to import from train.py
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the custom layer and functions from train.py
from train import FeatureExtractorLayer, load_model, CSV_PATH

# Load breed names
@st.cache_data
def load_breeds():
    import pandas as pd
    try:
        df = pd.read_csv(CSV_PATH)
        breeds = sorted(df["breed"].unique())
        return breeds
    except Exception as e:
        st.error(f"Error loading breed labels: {e}")
        return []

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make prediction
def predict_breed(model, image, breeds):
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_probs = predictions[0][top_3_indices]
    top_3_breeds = [breeds[i] for i in top_3_indices]
    return list(zip(top_3_breeds, top_3_probs))

# Main app
def main():
    st.title("Dog Breed Classifier")
    st.write("Upload an image of a dog to identify its breed")
    
    # Load model and breeds
    model = load_model()
    breeds = load_breeds()
    
    if model is None or not breeds:
        st.error("Could not initialize the application. Please check if model and label files exist.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction when button is clicked
        if st.button("Predict Breed"):
            with st.spinner("Analyzing image..."):
                results = predict_breed(model, image, breeds)
                
                # Display results
                st.subheader("Top 3 Predictions:")
                
                # Create a nice display for predictions
                col1, col2 = st.columns(2)
                
                for i, (breed, prob) in enumerate(results, 1):
                    # Format the breed name for better display
                    formatted_breed = breed.replace("_", " ").title()
                    
                    # Display prediction with progress bar
                    col1.write(f"{i}. {formatted_breed}")
                    col2.progress(float(prob))
                    col2.write(f"{prob*100:.2f}%")
                    
                # Display the most likely breed
                st.success(f"This dog most likely belongs to the {results[0][0].replace('_', ' ').title()} breed!")

if __name__ == "__main__":
    main()

# import tensorflow as tf
# import tensorflow_hub as hub
# import streamlit as st
# import numpy as np
# from PIL import Image
# import os
# import pandas as pd
# import random

# # FeatureExtractor Layer
# class FeatureExtractorLayer(tf.keras.layers.Layer):
#     def __init__(self, feature_extractor_url=None, trainable=False, **kwargs):
#         super(FeatureExtractorLayer, self).__init__(**kwargs)
#         self.feature_extractor_url = feature_extractor_url
#         self.trainable_setting = trainable

#         if feature_extractor_url:
#             self.feature_extractor = hub.KerasLayer(
#                 feature_extractor_url,
#                 trainable=trainable
#             )
#         else:
#             self.feature_extractor = None

#     def build(self, input_shape):
#         if self.feature_extractor is None and self.feature_extractor_url:
#             self.feature_extractor = hub.KerasLayer(
#                 self.feature_extractor_url,
#                 trainable=self.trainable_setting
#             )
#         super().build(input_shape)

#     def call(self, inputs):
#         if self.feature_extractor is None:
#             raise ValueError("Feature extractor not initialized")
#         return self.feature_extractor(inputs)

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             'feature_extractor_url': self.feature_extractor_url,
#             'trainable': self.trainable_setting
#         })
#         return config

#     @classmethod                         #@classmethod is used so that TensorFlow can recreate the custom layer from its config without needing an existing object.
#     def from_config(cls, config):
#         url = config.pop('feature_extractor_url', None)
#         trainable = config.pop('trainable', False)
#         return cls(feature_extractor_url=url, trainable=trainable, **config)

# # Load Model
# @st.cache_resource
# def load_model():
#     custom_objects = {
#         "FeatureExtractorLayer": FeatureExtractorLayer
#     }
#     model_path = os.path.join("model", "functionalapi_model.h5")
#     model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
#     return model

# # Load Breeds
# @st.cache_data         #makes app faster
# def load_breeds():
#     df = pd.read_csv("labels.csv")
#     breeds = sorted(df["breed"].unique())
#     return breeds


# labels_df = pd.read_csv("labels.csv")

# # Preprocess Image
# def preprocess_image(image):
#     IMG_SIZE = 224
#     image = image.resize((IMG_SIZE, IMG_SIZE))
#     image = np.array(image) / 255.0
#     return np.expand_dims(image, axis=0)

# # Show Breed Image
# def show_breed_image(predicted_breed):
#     breed_images = labels_df[labels_df['breed'] == predicted_breed]
#     if not breed_images.empty:
#         image_id = random.choice(breed_images['id'].tolist())
#         image_path = os.path.join("train", f"{image_id}.jpg")
#         if os.path.exists(image_path):
#             img = Image.open(image_path)
#             st.image(img, caption=f"Sample image of {predicted_breed}", use_container_width=False, width=300)
#         else:
#             st.warning(f"Image not found for: {predicted_breed}")
#     else:
#         st.warning(f"No images found for: {predicted_breed}")

# # --------------------- Streamlit UI --------------------- #
# st.set_page_config(page_title="Dog Breed Classifier", layout="centered")
# st.title("🐶 Dog Vision Prediction")
# st.write("""
# Upload a photo of a dog, and this app will predict the breed using a deep learning model trained on MobileNetV2.
# """)

# # Sidebar Info
# with st.sidebar:
#     st.header("About")
#     st.write("""
#     This web app uses a deep learning model to identify dog breeds from uploaded images.
    
#     It supports the classification of dozens of popular breeds, such as:
#     - Labrador Retriever
#     - German Shepherd
#     - Beagle
#     - Poodle
#     - Bulldog
#     - Chihuahua
#     - ...and many more!
#     """)

#     st.header("How it works")
#     st.write("""
#     1. Upload an image of a dog.\n
#     2. The model analyzes the image using a CNN.\n
#     3. You'll see the **top 3 predicted breeds** with confidence scores and example images.
#     """)

# # Load model and breeds
# model = load_model()
# breeds = load_breeds()

# # Upload image
# uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption='Uploaded Image', use_container_width=False, width=300)

#     # Predict
#     processed_image = preprocess_image(image)
#     prediction = model.predict(processed_image)
#     top_indices = np.argsort(prediction[0])[-3:][::-1]

#     # Display predictions
#     st.write("## 🔍 Predictions and Sample Images:")
#     for i, idx in enumerate(top_indices):
#         breed_name = breeds[idx]
#         confidence = prediction[0][idx] * 100
#         st.write(f"### {i+1}. {breed_name} — {confidence:.2f}% confidence")
#         show_breed_image(breed_name)
