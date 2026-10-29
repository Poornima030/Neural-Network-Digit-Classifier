"""Streamlit UI for the from-scratch NumPy digit classifier."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from model import NeuralNetwork
from preprocess import drawing_to_features, is_empty_drawing
from train import DEFAULT_WEIGHTS

st.set_page_config(page_title="Neural Network Digit Classifier", page_icon="🧠", layout="centered")


@st.cache_resource
def load_network(path: Path) -> NeuralNetwork:
    if not path.exists():
        raise FileNotFoundError(
            f"Trained weights not found at {path}. Run `python train.py` first."
        )
    return NeuralNetwork.load(path)


st.title("Neural Network Digit Classifier")
st.write(
    "Draw a digit from 0 to 9. A three-layer neural network implemented in NumPy "
    "(784 → 128 → 64 → 10) will classify it."
)

try:
    model = load_network(DEFAULT_WEIGHTS)
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

st.caption("Use white ink on the black canvas, similar to MNIST digits.")
canvas_kwargs = {
    "fill_color": "black",
    "stroke_width": 16,
    "stroke_color": "#FFFFFF",
    "background_color": "#000000",
    "width": 280,
    "height": 280,
    "drawing_mode": "freedraw",
    "key": "digit_canvas",
}
# Newer streamlit-drawable-canvas versions raise RuntimeError unless this is True.
if "return_image_data" in inspect.signature(st_canvas).parameters:
    canvas_kwargs["return_image_data"] = True

canvas_result = st_canvas(**canvas_kwargs)

if st.button("Predict digit", type="primary"):
    image_data = None
    try:
        image_data = canvas_result.image_data
    except RuntimeError:
        st.error(
            "The drawing canvas did not return image pixels. "
            "This app expects streamlit-drawable-canvas with return_image_data support."
        )
        st.stop()

    if is_empty_drawing(image_data, canvas_result.json_data):
        st.warning("Please draw a digit before predicting.")
    else:
        features = drawing_to_features(image_data)
        probabilities = model.predict_proba(features)[0]
        predicted = int(np.argmax(probabilities))
        st.subheader(f"Predicted digit: {predicted}")
        st.write("Class probabilities")
        st.bar_chart({str(digit): float(probabilities[digit]) for digit in range(10)})
        st.caption("The tallest bar is the model's chosen digit.")

st.markdown("---")
st.caption(
    "Forward pass, loss, and training use NumPy only — not TensorFlow, Keras, or PyTorch."
)
