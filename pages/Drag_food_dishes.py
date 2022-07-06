import streamlit as st
import numpy as np
import pandas as pd
import cv2
from skimage import io
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
    
st.header("🍩 Upload Food Dishes of your choice!")

def upload_image():
    uploaded = st.file_uploader("Upload Image")
    if uploaded is not None:
        
        display_image = Image.open(uploaded)
        st.image(display_image)
    return uploaded


def detect_image(food):
    food_model = hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/food_V1/1')
    
    # label dataset
    labelmap_url = "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_food_V1_labelmap.csv"
    input_shape = (224, 224)

    dish = np.asarray(io.imread(food), dtype="float")
    dish = cv2.resize(dish, dsize=input_shape, interpolation=cv2.INTER_CUBIC)
        
    # Scale values to [0, 1].
    dish = dish / dish.max()
        
    # The model expects an input of (?, 224, 224, 3).
    images = np.expand_dims(dish, 0)
    output = food_model(images)
    predicted_index = output.numpy().argmax()
    classes = list(pd.read_csv(labelmap_url)["name"])
    results = classes[predicted_index]
    st.subheader('🌟 Predicts')
    return st.success(results)


uploads = upload_image()
detect_image(uploads)

st.info('🚧 Some beautiful data visualizations will be added soon :)')