import streamlit as st
import folium
from PIL import Image
import numpy as np
import tensorflow as tf
import torch

# Load your model
model_path = "fake_model_streamlit.py"
model = FakePretrainedModel()
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model in evaluation mode

@st.cache(suppress_st_warning=True)

def predict_with_model(image_array):
    # Make a prediction using your CNN model
    prediction = model.predict(image_array)
    return prediction

def main():
    st.title("Latitude and Longitude to Map")

    # Input fields for latitude and longitude
    latitude = st.text_input("Enter Latitude:")
    longitude = st.text_input("Enter Longitude:")

    # Check if latitude and longitude are provided
    if latitude and longitude:
        try:
            # Parse latitude and longitude as floats
            latitude = float(latitude)
            longitude = float(longitude)

            # Create a Folium map centered at the specified coordinates
            m = folium.Map(location=[latitude, longitude], zoom_start=12)

            # Add a marker at the specified coordinates
            folium.Marker([latitude, longitude], popup=f"Lat: {latitude}, Lon: {longitude}").add_to(m)

            # Display the map
            st.write(m)

        except ValueError:
            st.error("Invalid input. Please enter valid latitude and longitude.")
        
            # File uploader for TIFF image
    uploaded_image = st.file_uploader("Upload TIFF Image", type=["tif", "tiff"], key="image")

    if uploaded_image is not None:
        st.write("Uploaded Image:")
        st.image(uploaded_image, use_column_width=True)

        # Process the uploaded image and make a prediction
        image = Image.open(uploaded_image)
        image = image.resize((224, 224))  # Resize the image to match your model's input size

        # Preprocess the image (you may need to adjust preprocessing based on your model)
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make a prediction using your CNN model
        prediction = predict_with_model(image_array)

        # Display the prediction result (you can customize this part)
        st.write(f"Prediction: {prediction[0][0]:.4f}")

if __name__ == "__main__":
    main()
