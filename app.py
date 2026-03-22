import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas

# Page config
st.set_page_config(page_title="Sketch Enhancer", layout="centered")

st.title("🎨 Sketch Enhancer & Artistic Generator")
st.write("Draw something below and see different artistic versions!")

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

# -------------------------------
# Image Processing Functions
# -------------------------------

def pencil_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    inv_blur = cv2.bitwise_not(blur)
    sketch = cv2.divide(gray, inv_blur, scale=256.0)
    return sketch

def cartoon_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def color_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return colored

def edge_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

# -------------------------------
# Main Logic
# -------------------------------

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype("uint8")

    # Check if user drew something
    if np.mean(img) < 5:
        st.warning("✏️ Please draw something first!")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        sketch = pencil_sketch(img)
        cartoon = cartoon_effect(img)
        color = color_map(img)
        edges = edge_detect(img)

        st.subheader("🖼️ Enhanced Outputs")

        col1, col2 = st.columns(2)
        with col1:
            st.image(sketch, caption="✏️ Pencil Sketch")
            st.image(edges, caption="🔍 Edge Detection")

        with col2:
            st.image(cartoon, caption="🎭 Cartoon Effect")
            st.image(color, caption="🎨 Color Map Art")
