import base64
import streamlit as st
import numpy as np
from PIL import ImageOps,Image
import cv2
def set_background(image_file):
    with open(image_file,"rb") as f:
        img_data=f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp{{
                background-image: url(data:image/png;base64,{b64_encoded});
                background-size: cover;
            }}
            </style>
        """
    st.markdown(style,unsafe_allow_html=True)
def convert_2_gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

def binarization(image):
    img, thresh = cv2.threshold(image, 0,255, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    return img, thresh

def dilate(image, words= False):
    img = image.copy()
    m = 3
    n = m - 2                   # n less than m for Vertical structuring element to dilate chars
    itrs = 4
    if words:
        m = 6
        n = m
        itrs = 3
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, m))
    dilation = cv2.dilate(img, rect_kernel, iterations = itrs)
    return dilation



def classify(image, model):
     #convert image to (224,224)
    image = ImageOps.fit(image,(32,32),Image.Resampling.LANCZOS)
    # convert image to numpy array
    image_array =np.array(image)
    # Normalize image
    normalized_image_array=(image_array.astype(np.float32)/127.5)-1 # Image pixel goes to -1 to 1

    # Set Model imput
    data=np.ndarray(shape=(1,32,32,3), dtype=np.float32) # 1 means uplaod only one image (224,224) is the size of the image and 3 means 3 channels
    # Make Prediction
    prediction = model.predict(data)

    # Post-process prediction (if needed)
    # Convert back to uint8 for displaying as an image
    return prediction  # Return the output image

# In your Streamlit app, you can use it like this:

