import streamlit as st
import numpy as np
import tensorflow as tf
import tf_keras
import tensorflow_hub as hub
from tf_keras.preprocessing.image import load_img, img_to_array
import os

# Constants
IMG_SIZE = 224

# Load model and label map
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "sequentialapi_model.h5")
    model = tf_keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    label_map = np.load('label_map.npy', allow_pickle=True).item()
    return model, label_map

# Preprocess uploaded image
def preprocess_uploaded_image(image):
    img = load_img(image, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Main app
def main():
    st.set_page_config(page_title="Dog Vision Prediction", layout="centered")
    st.title("🐶 Dog Vision Prediction")
    st.markdown("Upload a dog image and I'll predict its breed!")

    model, label_map = load_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        st.write("Classifying...")

        img_array = preprocess_uploaded_image(uploaded_file)
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_breed = label_map[predicted_index]
        confidence = predictions[0][predicted_index] * 100

        st.success(f"**Predicted Breed:** {predicted_breed}")
        st.info(f"**Confidence:** {confidence:.2f}%")

if __name__ == '__main__':
    main()
