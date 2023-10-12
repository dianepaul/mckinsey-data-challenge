import streamlit as st
from PIL import Image
from fake_model_streamlit import FakePretrainedModel
import random
import pandas as pd
import warnings


def predict_with_model(image_array):
    # Make a prediction using your CNN model
    model = FakePretrainedModel(image_array)
    prediction = model.predict()
    return prediction  # random.random()


def page_plume_detection():
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
        prediction = predict_with_model(image_sat)
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


def page_model_understanding():
    st.subheader("Model Understanding")
    training_data = pd.read_csv("metadata_train.csv")
    training_data = training_data[["date", "plume", "lat", "lon", "coord_x", "coord_y"]]
    training_data["color"] = [
        "#7BA399" if plume == "yes" else "#EE6C4D" for plume in training_data["plume"]
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


st.set_page_config(
    page_title="Plume detection",
    page_icon="🪶",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "# This page can be used to upload an image and detect a potential methane plume, please use Tif 64x64 format"
    },
)
warnings.filterwarnings("ignore")

# Load custom CSS
with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center'>Plume detection</h1>", unsafe_allow_html=True
)

selected_page = st.sidebar.selectbox(
    "Go to Page", ["Plume Detection", "Model Understanding"]
)

if selected_page == "Plume Detection":
    page_plume_detection()
elif selected_page == "Model Understanding":
    page_model_understanding()
