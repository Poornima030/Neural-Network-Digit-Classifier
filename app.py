import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from streamlit_drawable_canvas import st_canvas
import os

st.set_page_config(page_title="Neural Network Digit Classifier", page_icon="🧠", layout="centered")

st.title("Neural Network Digit Classifier")
st.write("Draw a digit (0–9) below, and the neural network will predict it!")

# ------------------------------
# 1️⃣ Train or Load Model
# ------------------------------
MODEL_PATH = "mnist_model.keras"  # use .keras format (safe for TF 2.18+)

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if not os.path.exists(MODEL_PATH):
    st.info("Training the model for the first time... Please wait ⏳")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = build_model()
    model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))
    model.save(MODEL_PATH)
    st.success("✅ Model trained and saved successfully!")
else:
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Pre-trained model loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {e}")
        st.info("Rebuilding and retraining the model...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        model = build_model()
        model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))
        model.save(MODEL_PATH)
        st.success("✅ Model rebuilt and saved successfully!")

# ------------------------------
# 2️⃣ Drawing Canvas
# ------------------------------
st.write("✏️ Draw your digit below:")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ------------------------------
# 3️⃣ Predict Button
# ------------------------------
if st.button("🔍 Predict Digit"):
    if canvas_result.image_data is not None:
        img = np.array(canvas_result.image_data)
        img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])  # grayscale
        img_gray = np.expand_dims(img_gray, axis=-1)  # (H, W, 1)
        img_resized = tf.image.resize(img_gray, (28, 28)).numpy().astype("float32") / 255.0

        prediction = model.predict(img_resized.reshape(1, 28, 28, 1))
        st.markdown(f"<h3 style='text-align:center;'>Predicted Digit: <span style='color:#00BFFF;'>{np.argmax(prediction)}</span></h3>", unsafe_allow_html=True)
    else:
        st.warning("Please draw something before predicting.")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and TensorFlow")
