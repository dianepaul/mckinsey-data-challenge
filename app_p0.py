import streamlit as st
import pandas as pd
from PIL import Image


class Streamlit_Page0:
    def __init__(self):
        self.metadata = pd.read_csv("metadata_test.csv", header=0)
        self.selected_id = None
        self.selected_date = None
        self.latitude = None
        self.longitude = None

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
