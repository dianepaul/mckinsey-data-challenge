import streamlit as st
from PIL import Image
from fake_model_streamlit import FakePretrainedModel
import pandas as pd


class Streamlit_Page1:
    def __init__(self):
        pass

    def predict_with_model(self, image_array):
        # Make a prediction using your CNN model
        model = FakePretrainedModel(image_array)
        prediction = model.predict()
        return prediction  # random.random()

    def page_plume_detection(self):
        st.subheader("Plume Detection")

        col1, col2 = st.columns([3, 5])

        # File uploader for TIFF image
        col1.header("Satellite image")
        uploaded_image = col1.file_uploader(
            "Upload TIFF Image", type=["tif", "tiff"], key="image"
        )

        if uploaded_image is not None:
            # Process the uploaded image and make a prediction
            image_sat = Image.open(uploaded_image)
            image_sat = image_sat.resize((64, 64))  # Resize the image to 64x64
            prediction = self.predict_with_model(image_sat)
            col1.header(f"Prediction: {prediction:.4f}")

            col1.write("Uploaded Image:")
            # col1.image(image_sat, caption="Uploaded Image", use_column_width=True)

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
                coord = pd.DataFrame({"lat": [latitude], "lon": [longitude]})

                # Display the map in a separate column
                col2.map(coord, zoom=12)

            except ValueError:
                st.error("Invalid input. Please enter valid latitude and longitude")
