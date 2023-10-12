# Import necessary libraries
import streamlit as st
import warnings
from app_p0 import Streamlit_Page0
from app_p1 import Streamlit_Page1
from app_p2 import Streamlit_Page2

# Configure Streamlit app settings
st.set_page_config(
    page_title="Plume detection",  # Set the page title
    page_icon="🪶",  # Set a custom page icon
    layout="wide",  # Use wide layout
    initial_sidebar_state="collapsed",  # Start with the sidebar collapsed
    menu_items={
        "About": "# This page can be used to upload an image and detect a potential methane plume, please use Tif 64x64 format"
    },  # Add an "About" section to the sidebar
)
warnings.filterwarnings("ignore")  # Ignore Python warnings

# Read and apply custom CSS styles from 'custom.css' file
with open("custom.css", "r") as f:
    custom_css = f.read()
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# Display the app title
st.markdown(
    "<h1 style='text-align: center'>Plume detection</h1>", unsafe_allow_html=True
)

# Sidebar: Select the desired page
selected_page = st.sidebar.selectbox(
    "Go to Page",
    ["Plume Detection From Test", "Plume Detection Local", "Model Understanding"],
)

# Navigate to the selected page
if selected_page == "Plume Detection From Test":
    Streamlit_Page0().page_plume_detection()
elif selected_page == "Plume Detection Local":
    Streamlit_Page1().page_plume_detection()
elif selected_page == "Model Understanding":
    Streamlit_Page2().page_model_understanding()
