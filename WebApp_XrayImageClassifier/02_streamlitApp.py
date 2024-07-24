import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# Load the saved model
last_model = tf.keras.models.load_model('model.h5')

# Streamlit Application
st.title('Chest X-ray Image Classifier')

def classify_image(image, model):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prediction = classify_image(uploaded_file, last_model)
    if prediction[0][0] >= 0.5:
        st.write("Prediction: COVID-19")
    else:
        st.write("Prediction: Normal")
