import streamlit as st
import pandas as pd
from PIL import Image
from model_streamlit import Model_results


class Streamlit_Page0:
    def __init__(self):
        # Initialize the Streamlit app and class attributes
        self.metadata = pd.read_csv("metadata_test.csv", header=0)
        self.selected_id = None
        self.selected_date = None
        self.latitude = None
        self.longitude = None

    def update_metadata(self):
        # Update the selected ID based on the selected date
        if self.selected_date:
            filtered_ids = self.metadata[self.metadata["date"] == self.selected_date][
                "id_coord"
            ].unique()
            self.selected_id = st.selectbox("Select ID", filtered_ids)
        else:
            # If no date is selected, allow selecting from unique IDs
            self.selected_id = st.selectbox(
                "Select ID", self.metadata["id_coord"].unique()
            )

    def page_plume_detection(self):
        # Create the Streamlit app page for plume detection

        # Select a date for plume detection
        self.selected_date = st.selectbox("Select Date", self.metadata["date"].unique())

        # Callback function for updating selected ID based on the date
        self.update_metadata()

        # Construct the file name for plume detection
        file_name = (
            str(self.selected_date)
            + "_methane_mixing_ratio_"
            + self.selected_id
            + ".tiff"
        )

        # Predict plume probability based on the selected image
        prediction = Model_results(file_name).predict()

        if prediction is not None:
            st.header(f"Prediction: {prediction:.4f}")

        if self.selected_id:
            # Get the corresponding latitude and longitude for the selected ID and date
            self.latitude = self.metadata.loc[
                (self.metadata["id_coord"] == self.selected_id)
                & (self.metadata["date"] == self.selected_date),
                "lat",
            ].values[0]
            self.longitude = self.metadata.loc[
                (self.metadata["id_coord"] == self.selected_id)
                & (self.metadata["date"] == self.selected_date),
                "lon",
            ].values[0]

            st.write(
                f"Geographical Coordinates Lat: {self.latitude:.4f}, Lon: {self.longitude:.4f}"
            )

            if st.button("Show Map"):
                st.map(pd.DataFrame({"lat": [self.latitude], "lon": [self.longitude]}))


if __name__ == "__main__":
    # Run the Streamlit app with the Page 0 for plume detection
    Streamlit_Page0().page_plume_detection()
