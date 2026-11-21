import streamlit as st
import numpy as np
import json
from PIL import Image


# ---------------------------
# Configuration
# ---------------------------
IMG_SIZE = 128

st.set_page_config(
    page_title="PlantDisease AI",
    page_icon="🌿",
    layout="wide"
)

# Diagnostic message
st.title("🌿 PlantDisease AI")
st.info("Application started. Checking TensorFlow...")

# ---------------------------
# Import TensorFlow
# ---------------------------
try:
    import tensorflow as tf
    st.success(f"TensorFlow imported successfully: {tf.__version__}")
except Exception as e:
    st.error("TensorFlow import failed!")
    st.exception(e)
    st.stop()

# ---------------------------
# Load model and class names
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "plant_disease_model.keras",
        compile=False
    )

@st.cache_data
def load_classes():
    with open("class_names.json", "r") as f:
        return json.load(f)

try:
    st.info("Loading the trained model...")
    model = load_model()
    st.success("Model loaded successfully!")

    class_names = load_classes()
    st.success("Class names loaded successfully!")

except Exception as e:
    st.error("An error occurred while loading the model or class names!")
    st.exception(e)
    st.stop()


# ---------------------------
# Interface styling
# ---------------------------
st.markdown("""
<style>
.main {
    background-color: #f6faf6;
}
h1 {
    color: #206b45;
}
.result-card {
    padding: 20px;
    border-radius: 15px;
    background-color: #edf7ef;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.title("🌿 PlantDisease AI")
st.subheader("Tomato Leaf Disease Detection")
st.write(
    "Upload a tomato leaf image and our deep learning model "
    "will predict its disease category."
)

st.divider()

# ---------------------------
# Image upload
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    # ---------------------------
    # Prediction
    # ---------------------------
    resized_image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(resized_image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    display_name = predicted_class.replace("Tomato___", "").replace("_", " ")

    with col2:
        st.markdown("### 🔍 Prediction Result")

        st.success(f"**Predicted Class: {display_name}**")

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(float(predictions[predicted_index]))

        # Confidence caution
        if confidence < 60:
            st.warning(
                "The model is uncertain about this prediction. "
                "Try uploading a clearer leaf image."
            )

    # ---------------------------
    # Probability chart
    # ---------------------------
    st.divider()
    st.subheader("📊 Class Probability Distribution")

    readable_names = [
        name.replace("Tomato___", "").replace("_", " ")
        for name in class_names
    ]

    probability_data = {
        "Disease": readable_names,
        "Probability (%)": predictions * 100
    }

    st.bar_chart(
        probability_data,
        x="Disease",
        y="Probability (%)"
    )

    # ---------------------------
    # Top 3 predictions
    # ---------------------------
    st.subheader("🏆 Top 3 Predictions")

    top_indices = np.argsort(predictions)[::-1][:3]

    for index in top_indices:
        name = readable_names[index]
        probability = predictions[index] * 100

        st.write(f"**{name}** — {probability:.2f}%")
        st.progress(float(predictions[index]))

# ---------------------------
# Footer
# ---------------------------
st.divider()
st.caption(
    "Academic Deep Learning Project | "
    "CNN Transfer Learning with MobileNetV2"
)