import streamlit as st
import numpy as np
import pandas as pd
import cv2
from skimage import io
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
    
st.header("🌭 Try out from some of the examples below!")

def example_image():
    example_1 = 'pages/images/spaghetti.jpeg'
    example_2 = 'pages/images/onam.jpg'
    example_3 = 'pages/images/hotdog.jpg'
    example_4 = 'pages/images/Japanese_Sushi.jpg'
    example_5 = 'pages/images/sweet.jpg'
    example_6 = 'pages/images/sweet_2.jpg'
    
    options = st.selectbox(
     'Select any one of the dish!😋',
     ('', example_1, example_2, example_3, example_4, example_5, example_6))

    if options == example_1:
        display = Image.open(example_1)
        st.image(display)
        return example_1
    elif options == example_2:
        display = Image.open(example_2)
        st.image(display)
        return example_2
        
    elif options == example_3:
        display = Image.open(example_3)
        st.image(display)
        return example_3
        
    elif options == example_4:
        display = Image.open(example_4)
        st.image(display)
        return example_4
        
    elif options == example_5:
        display = Image.open(example_5)
        st.image(display)
        return example_5

    elif options == example_6:
        display = Image.open(example_6)
        st.image(display)
        return example_6

    else:
        st.write('Please select an image from examples or upload one! 👆')

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


examples = example_image()
detect_image(examples)

#st.info('🚧 Some beautiful data visualizations will be added soon :)')
