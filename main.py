import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify, set_background
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import json
from streamlit_lottie import st_lottie

# Sklearn package -> function
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report

# Tensorflow packages


#set_background('Handwritten-character-recognition logo.png')
#Set Title
st.title("Decode")
#Set Header
st.header("Please Upload an Image ")
#Uplaod a File
file = st.file_uploader('',type=['jpeg', 'jpg', 'png'])
st.write('Click on Browse files to locate the file')
# Load Classifier
def load_lottieurl(url: str):
    r=requests.get(url)
    if(r.status_code !=200):
        return None
    return r.json()
lottie_hello =load_lottieurl("https://lottie.host/b11351d2-a03c-4e09-87ae-c4ae34e9a2c2/eBQWf1QS6Q.json")
st_lottie(lottie_hello,key="hello")
model=load_model('DenseNet121_model.h5')
#load class names
#with open('labels.txt','r') as f:
#    class_names=[a[:-1].split(' ')[1] for a in f.readlines()]
#    f.close()'''
#display image

if file is not None:
    image= Image.open(file).convert('RGB')
    hide_animation_style = """
            <style>
            .css-1l08gxm-AnimationWrapper {animation: none !important;}
            </style>
            """

    st.markdown(hide_animation_style, unsafe_allow_html=True)
    st.image(image, use_column_width=True)

    out= classify(image, model)
    st.write(out)

#    st.write("##{}".format(class_name))
#    st.write("## score: {}".format(conf_score))
#    image1= Image.open(file).convert('RGB')
#    st.image(image1)
#    Classify image
    
    