# Neural Network Digit Classifier

This project is a simple, yet powerful implementation of a Convolutional Neural Network (CNN) trained to classify handwritten digits (0-9) from the famous **MNIST dataset**.

The classifier is built using popular machine learning libraries and is deployed as an interactive web application using Streamlit, allowing users to draw a digit and instantly see the model's prediction.

## 🚀 Live Demo

Experience the classifier live without needing to install anything:

**[Try the Streamlit App Here](https://neural-network-digit-classifier-h6wrpoxp7rxtxuqwez7ev5.streamlit.app/)**

## 💡 Features

* **Handwritten Digit Recognition:** Classifies digits from 0 to 9.
* **Interactive Drawing Canvas:** Allows users to draw a digit directly on the web interface.
* **Streamlit Web Interface:** Provides an easy-to-use, real-time prediction environment.
* **Trained CNN Model:** Utilizes a Convolutional Neural Network (likely built with TensorFlow/Keras or PyTorch) for high accuracy.

## 🛠️ Installation and Setup

To run this project locally, follow these steps:

### Prerequisites

You need Python 3.8+ installed on your system.

### 1. Clone the repository

```bash
git clone [https://github.com/Poornima030/Neural-Network-Digit-Classifier.git](https://github.com/Poornima030/Neural-Network-Digit-Classifier.git)
cd Neural-Network-Digit-Classifier
````

### 2\. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
.\venv\Scripts\activate   # On Windows
```

### 3\. Install Dependencies

Install all required Python packages (Streamlit, TensorFlow/Keras, NumPy, etc.) from the `requirements.txt` file (assuming it exists in your repository).

```bash
pip install -r requirements.txt
```

*(Note: If you do not have a `requirements.txt`, you will need to install `streamlit` and the ML library used, e.g., `pip install streamlit tensorflow numpy`)*

## 🏃 How to Run the App

Assuming the main application file is named `app.py` or `digit_classifier.py`:

```bash
streamlit run app.py
```

The command will automatically open the application in your default web browser (usually at `http://localhost:8501`).

## 📁 Repository Structure

The key files in this repository are expected to be:

```
Neural-Network-Digit-Classifier/
├── app.py                      # Main Streamlit application script
├── model.h5 / model.pth        # The trained Neural Network model file
├── README.md                   # This file
└── requirements.txt            # Project dependencies
```

## 🧠 Technical Details

This project demonstrates the core concepts of Neural Networks, specifically:

  * **Model Type:** Convolutional Neural Network (CNN).
  * **Dataset:** MNIST (Modified National Institute of Standards and Technology database) of handwritten digits.
  * **Libraries:** Likely built with **Streamlit** for the frontend, and **TensorFlow** or **PyTorch** for the model training and prediction.
  * **Concepts Demonstrated:** Image preprocessing, model loading, and real-time inference.
