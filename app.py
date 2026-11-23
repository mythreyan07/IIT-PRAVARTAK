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

PAGES = {
    "Image Preprocessing": "frontend.preprocessing_page",
}

st.sidebar.title("EAGLE-DAR")
st.sidebar.caption("Disaster Assessment and Response")
page = st.sidebar.radio("Navigation", list(PAGES.keys()))

if page == "Image Preprocessing":
    from frontend.preprocessing_page import render_preprocessing_page
    render_preprocessing_page()
