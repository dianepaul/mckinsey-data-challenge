import streamlit as st
import pandas as pd
from PIL import Image
from model_streamlit import Model_results


class Streamlit_Page0:
    def __init__(self):
        self.metadata = pd.read_csv("metadata_test.csv", header=0)
        self.selected_id = None
        self.selected_date = None
        self.latitude = None
        self.longitude = None

    def predict_with_model(self, image):
        # Make a prediction using your CNN model
        model = Model_results(image)
        prediction = model.predict()
        return prediction

    def update_metadata(self):
        # Add an optional filter for IDs based on the selected date
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
        self.selected_date = st.selectbox("Select Date", self.metadata["date"].unique())

        # Callback function for updating selected ID
        self.update_metadata()
        file_name = (
            str(self.selected_date)
            + "_methane_mixing_ratio_"
            + self.selected_id
            + ".tiff"
        )
        prediction = self.predict_with_model(file_name)
        st.header(f"Prediction: {prediction:.4f}")

        if self.selected_id:
            # Get the corresponding latitude and longitude
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
                f"Geographical Coordinates Lat: {self.latitude}, Lon: {self.longitude}"
            )

            if st.button("Show Map"):
                st.map(pd.DataFrame({"lat": [self.latitude], "lon": [self.longitude]}))

            # Now you can use self.latitude and self.longitude to update the map or display other information.


if __name__ == "__main__":
    Streamlit_Page0().page_plume_detection()
