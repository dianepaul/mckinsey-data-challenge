import streamlit as st
import warnings
from app_p0 import Streamlit_Page0
from app_p1 import Streamlit_Page1
from app_p2 import Streamlit_Page2


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
    "Go to Page",
    ["Plume Detection From Test", "Plume Detection Local", "Model Understanding"],
)


if selected_page == "Plume Detection From Test":
    Streamlit_Page0().page_plume_detection()
elif selected_page == "Plume Detection Local":
    Streamlit_Page1().page_plume_detection()
elif selected_page == "Model Understanding":
    Streamlit_Page2().page_model_understanding()
