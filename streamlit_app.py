import streamlit as st

st.header('🌮🍜 FOOD DETECTION APP 🍕🍰')

st.markdown("""
This App helps to identify different types of food dishes 😋\n

✨ Some valid info -->\n
  - It uses a pre-trained model from **Tensorflow Hub**. This model is based on MobileNet V1.\n
  - It can recognize 2023 food dishes from images.\n
  - The training set includes entrees, side dishes, desserts, snacks, etc.\n
  - A non-food image or the dish image is not well-cropped, then the output of the model may be meaningless. 🤷‍♀️\n
  - This model cannot determine whether the dish is edible or not.""")
    

st.subheader('👈 Go Ahead and try out this app using the sidebar provided!')