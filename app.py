import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
import pickle

st.title("🎨 Smart Sketch Recognizer")

# Load trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Labels (example classes)
labels = ["Cat", "Tree", "House", "Car", "Star"]

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
    return gray, edges, colored

def predict(img):
    img = cv2.resize(img, (28, 28))
    img = img.flatten().reshape(1, -1)
    pred = model.predict(img)[0]
    return labels[pred]

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    gray, edges, result = process_image(img)

    prediction = predict(gray)

    st.subheader("🧠 Prediction:")
    st.success(f"👉 It looks like: {prediction}")

    col1, col2 = st.columns(2)

    with col1:
        st.image(edges, caption="Edges")

    with col2:
        st.image(result, caption="Stylized Image")
