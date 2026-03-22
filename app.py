import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
from sklearn import svm

# Page config
st.set_page_config(page_title="Sketch Recognizer", layout="centered")

st.title("🎨 Smart Sketch Recognizer")
st.write("Draw something below 👇 and AI will guess it!")

# -------------------------------
# Load / Create Model (No file needed)
# -------------------------------
@st.cache_resource
def load_model():
    # Dummy dataset (replace with real later)
    X = np.random.rand(100, 784)
    y = np.random.randint(0, 5, 100)

    model = svm.SVC(probability=True)
    model.fit(X, y)

    return model

model = load_model()

# Labels
labels = ["Cat 🐱", "Tree 🌳", "House 🏠", "Car 🚗", "Star ⭐"]

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
# Image Processing
# -------------------------------
def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_inv = cv2.bitwise_not(edges)
    colored = cv2.applyColorMap(edges_inv, cv2.COLORMAP_JET)
    return gray, edges, colored

# -------------------------------
# Prediction Function
# -------------------------------
def predict(img):
    img = cv2.resize(img, (28, 28))
    img = img.flatten().reshape(1, -1)

    pred = model.predict(img)[0]
    prob = model.predict_proba(img)[0]

    return labels[pred], np.max(prob)

# -------------------------------
# Run when user draws
# -------------------------------
if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype("uint8")
    
    # Check if user actually drew something
    if np.sum(img) == 0:
        st.warning("✏️ Please draw something first!")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        gray, edges, result = process_image(img)

        label, confidence = predict(gray)

        st.subheader("🧠 Prediction")
        st.success(f"👉 {label}")
        st.info(f"Confidence: {confidence:.2f}")

        col1, col2 = st.columns(2)

        with col1:
            st.image(edges, caption="Sketch (Edges)")

        with col2:
            st.image(result, caption="Stylized Image")
