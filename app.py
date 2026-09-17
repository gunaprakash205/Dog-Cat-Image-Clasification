import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("cat_dog_cnn.keras")

# App title
st.title("🐱 Cat vs Dog Image Classifier")

st.write(
    "Upload an image and the CNN will predict whether it is a Cat or Dog."
)

# Important limitation
st.warning(
    "⚠️ This model is trained only for Cat vs Dog classification. "
    "Predictions for other types of images are not reliable."
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload a Cat or Dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize
    image_resized = image.resize((64, 64))

    # Convert to NumPy
    image_array = np.array(image_resized)

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array, verbose=0)[0][0]

    if prediction >= 0.5:
        predicted_class = "Dog 🐶"
        confidence = prediction
    else:
        predicted_class = "Cat 🐱"
        confidence = 1 - prediction

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2%}")