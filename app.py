import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import torch
from fake_model_streamlit import FakePretrainedModel
import random 
import pandas as pd

st.set_page_config(
    page_title= "Plume detection",
    layout="wide", 
    initial_sidebar_state="collapsed",
    menu_items = {"About": "# This page can be used to upload an image and detect a potential methane plume, please use Tif 64x64 format"}, 
    )

# @st.cache(suppress_st_warning=True)
import warnings
warnings.filterwarnings("ignore")

# Load custom CSS
with open("custom.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center'>Plume detection</h1>", unsafe_allow_html=True)

def predict_with_model(image_array):
    # Make a prediction using your CNN model
    # prediction = FakePretrainedModel.predict(image_array)
    return random.random() # prediction

def main():
    col1, col2 = st.columns([3, 4])

    # File uploader for TIFF image
    col1.header("Satellite image")
    uploaded_image = col1.file_uploader("Upload TIFF Image", type=["tif", "tiff"], key="image")
    
    if uploaded_image is not None:
         # Process the uploaded image and make a prediction
        image_sat = Image.open(uploaded_image)
        image_sat = image_sat.resize((64, 64))  # Resize the image to 64x64
        prediction = predict_with_model(image_sat)
        col1.header(f"Prediction: {prediction:.4f}")
        
        col1.write("Uploaded Image:")   
        # col1.image(image, caption="Uploaded Image", use_column_width=True)
        col1.image(image_sat)

        # image_array = np.array(image) / 255.0  # Normalize pixel values
        # image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
    col2.header("Latitude and Longitude to Map")
    # Input fields for latitude and longitude
    latitude = col2.text_input("Enter Latitude:")
    longitude = col2.text_input("Enter Longitude:")

    # Check if latitude and longitude are provided
    if latitude and longitude:
        try:
            # Parse latitude and longitude as floats
            latitude = float(latitude)
            longitude = float(longitude)
            coord = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
            
            # Display the map in a separate column
            col2.map(coord, zoom=12)

        except ValueError:
            st.error("Invalid input. Please enter valid latitude and longitude.")

if __name__ == "__main__":
    main()