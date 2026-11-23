"""
EAGLE-DAR Streamlit Application Entry Point

Run with:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="EAGLE-DAR",
    page_icon="🛰️",
    layout="wide",
)

PAGES = [
    "Dataset Inspection",
    "Image Preprocessing",
    "Model Training",
    "Damage Assessment",
    "Reports",
]

st.sidebar.title("EAGLE-DAR")
st.sidebar.caption("Disaster Assessment and Response")
page = st.sidebar.radio("Navigation", PAGES)

if page == "Dataset Inspection":
    from frontend.dataset_page import render_dataset_page
    render_dataset_page()
elif page == "Image Preprocessing":
    from frontend.preprocessing_page import render_preprocessing_page
    render_preprocessing_page()
elif page == "Model Training":
    from frontend.training_page import render_training_page
    render_training_page()
elif page == "Damage Assessment":
    from frontend.inference_page import render_inference_page
    render_inference_page()
elif page == "Reports":
    from frontend.report_page import render_report_page
    render_report_page()
