import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Sketch Enhancer", layout="centered")

st.title("🎨 Sketch Enhancer & Artistic Generator")
st.write("Draw something below 👇 and explore different artistic styles!")

# -------------------------------
# Canvas
# -------------------------------
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
# Image Effects Functions
# -------------------------------

# ✏️ Pencil Sketch
def pencil_sketch(img):
    gray, sketch = cv2.pencilSketch(
        img, sigma_s=60, sigma_r=0.07, shade_factor=0.05
    )
    return sketch

# 🎭 Cartoon Effect (Improved)
def cartoon_effect(img):
    color = cv2.bilateralFilter(img, d=9, sigmaColor=250, sigmaSpace=250)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        10,
    )

    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

# 🎨 Heatmap Effect (Better Color Map)
def heatmap_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return colored

# 🖼️ Oil Painting Effect
def oil_painting(img):
    return cv2.stylization(img, sigma_s=60, sigma_r=0.6)

# 🔍 Edge Detection
def edge_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 50, 150)

# -------------------------------
# Main Logic
# -------------------------------
if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype("uint8")

    # Check if user actually drew something
    if np.mean(img) < 5:
        st.warning("✏️ Please draw something first!")
    else:
        # Convert RGBA → BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        # Apply Effects
        sketch = pencil_sketch(img)
        cartoon = cartoon_effect(img)
        heatmap = heatmap_effect(img)
        oil = oil_painting(img)
        edges = edge_detect(img)

        # -------------------------------
        # Display Results
        # -------------------------------
        st.subheader("🖼️ Enhanced Outputs")

        col1, col2 = st.columns(2)

        with col1:
            st.image(sketch, caption="✏️ Pencil Sketch", use_column_width=True)
            st.image(cartoon, caption="🎭 Cartoon Effect", use_column_width=True)

        with col2:
            st.image(heatmap, caption="🎨 Heatmap Effect", use_column_width=True)
            st.image(oil, caption="🖼️ Oil Painting", use_column_width=True)

        st.image(edges, caption="🔍 Edge Detection", use_column_width=True)
