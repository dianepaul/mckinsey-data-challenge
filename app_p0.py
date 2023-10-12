import streamlit as st
import pandas as pd
from fake_model_streamlit import FakePretrainedModel
from PIL import Image


class Streamlit_Page0:
    def __init__(self):
        self.metadata = pd.read_csv("metadata_test.csv", header=0)
        self.selected_id = None
        self.latitude = None
        self.longitude = None
        self.image_filename = None

    def predict_with_model(self, image_array):
        # Make a prediction using your CNN model
        model = FakePretrainedModel(image_array)
        prediction = model.predict()
        return prediction  # random.random()

    def page_plume_detection(self):
        st.subheader("Plume Detection")
        # Create a selection box for Date
        selected_date = st.selectbox("Select Date", self.metadata["date"].unique())

        # Add an optional filter for IDs based on the selected date
        if selected_date:
            filtered_ids = self.metadata[self.metadata["date"] == selected_date][
                "id_coord"
            ].unique()
            self.selected_id = st.selectbox("Select ID", filtered_ids)
        else:
            # If no date is selected, allow selecting from unique IDs
            self.selected_id = st.selectbox(
                "Select ID", self.metadata["id_coord"].unique()
            )
        st.write(self.selected_id)
        selected_date = self.metadata.loc[
            self.metadata["id_coord"] == self.selected_id, "date"
        ].values[0]
        st.write(selected_date)

        self.image_filename = (
            str(selected_date) + "_methane_mixing_ratio_" + self.selected_id + ".tif"
        )
        st.write(self.image_filename)

        col1, col2 = st.columns([3, 5])

        # File uploader for TIFF image
        col1.header("Satellite image")
        image = Image.open("test_data/" + self.image_filename)

        if image is not None:
            # Process the uploaded image and make a prediction
            image_sat = Image.open(image)
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
