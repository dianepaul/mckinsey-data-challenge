import streamlit as st
import pandas as pd


class Streamlit_Page2:
    def __init__(self):
        pass

    def page_model_understanding(self):
        st.subheader("Model Understanding")
        training_data = pd.read_csv("metadata_train.csv")
        training_data = training_data[
            ["date", "plume", "lat", "lon", "coord_x", "coord_y"]
        ]
        training_data["color"] = [
            "#7BA399" if plume == "yes" else "#EE6C4D"
            for plume in training_data["plume"]
        ]
        st.write(
            f"You can visualize hereafter the distribution of plumes from the training dateset used to build the model around the world. This dataset contanins {len(training_data)} points."
        )

        # Create a custom legend using an HTML <div> element
        # Create a custom legend using an HTML <div> element
        legend_html = """
        <div style="position: absolute; top: 10px; left: 10px; background-color: white; padding: 10px; border: 0.5px solid lightgray; border-radius: 5px;">
            <p style="margin: 0;"></p>
            <p style="margin: 0;"><span style="background-color: #C3D5D0; width: 20px; height: 20px; display: inline-block; margin-right: 5px;"></span>Plume</p>
            <p style="margin: 0;"><span style="background-color: #EE6C4D; width: 20px; height: 20px; display: inline-block; margin-right: 5px;"></span>No Plume</p>
        </div>
        """
        st.map(training_data, color="color")

        # Display the custom legend using st.markdown
        st.markdown(legend_html, unsafe_allow_html=True)
        # st.write(m)
