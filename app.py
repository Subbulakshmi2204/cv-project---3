import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Sketch to Image", layout="centered")

st.title("🎨 Sketch to Image Converter")
st.write("Draw something below 👇")

# Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=5,
    stroke_color="white",
    background_color="black",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_inv = cv2.bitwise_not(edges)
    colored = cv2.applyColorMap(edges_inv, cv2.COLORMAP_JET)
    return edges, colored

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    edges, result = process_image(img)

    st.subheader("🖼️ Results")
    col1, col2 = st.columns(2)

    with col1:
        st.image(edges, caption="Edges")

    with col2:
        st.image(result, caption="Stylized Image")
