import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Load model
model = load_model("mnist_model.h5")

st.title("🧠 Handwritten Digit Recognition")
st.write("Upload a 0–9 handwritten digit to predict using the trained MNIST model.")

uploaded_file = st.file_uploader("📁 Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Step 1: Load and show image
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Uploaded Image", width=150)

    # Step 2: Auto-invert if background is white
    if np.array(image).mean() > 127:
        image = ImageOps.invert(image)

    # Step 3: Resize and preprocess
    image = image.resize((28, 28))
    img = np.array(image).astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Step 4: Predict
    prediction = model.predict(img)
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"🧾 Predicted Digit: **{pred_class}** with confidence **{confidence*100:.2f}%**")

    # Step 5: Plot probability graph
    st.write("### Prediction Probabilities")
    fig, ax = plt.subplots()
    ax.bar(range(10), prediction[0])
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_title("Model Confidence for Each Digit")
    st.pyplot(fig)
