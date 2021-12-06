import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2

import keras
from keras.models import load_model
from scipy.spatial import distance
import scipy


from skimage.transform import resize
#import time


# start on terminal
# streamlit run app.py


################
##    Model   ##
################


# Load the model
model = load_model('face_mask_detector.h5')
# Load the cascade
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



################
##   Title    ##
################
# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(page_title='Face Mask Detector', page_icon='😷', layout='centered', initial_sidebar_state='expanded')

# Designing the interface

image = Image.open('backgrounds.png')


col1, col2, col3= st.columns([80,2,8])
with col1:
    st.image(image)
    st.write("")
with col2:
    #st.title('Face Mask Detector App')
    #st.subheader('Face Mask Detector web app to predictwheather the face in the image with mask , No Mask , Weared mask incorrect')
    #st.write('Face Mask Detector web app to predictwheather the face in the image with mask , No Mask , Weared mask incorrectly')
    st.write("")
with col3:
    st.write("")
    #st.write('Face Mask Detector web app to predictwheather the face in the image with mask , No Mask , Weared mask incorrectly')




st.title("Face Mask Detector App")
st.write("A deep learning app to predict faces with Mask , without mask and weared mask incorrectly.")


################
##    Image   ##
################


def import_and_predict(image_data, model):

        size = (150,150)    #
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(150, 150),    interpolation=cv2.INTER_CUBIC))/255.

        img_reshape = img_resize[np.newaxis,...]

        prediction = model.predict(img_reshape)

        return prediction

#############################


###########################
##    Facial Detection   ##  TBD
###########################



#############################
uploaded_file = st.file_uploader("⬇ Upload the Face image here ⬇ ", type=['jpg','png', 'jpeg'])


if uploaded_file is not None:

    u_img = Image.open(uploaded_file)
    #st.success("Success")
    st.image(u_img, 'Uploaded Image', use_column_width=True)
 
    prediction = import_and_predict(u_img, model)

    if np.argmax(prediction) == 0:
        #st.subheader("Wearing a Mask")
        m1 = '<p style="color:Blue; font-size: 30px;">Wearing Mask Incorrectly! </p>'
        st.markdown(m1, unsafe_allow_html=True)
    elif np.argmax(prediction) == 1:
        m2 = '<p style="color:Blue; font-size: 30px;">Wearing a Mask</p>'
        st.markdown(m2, unsafe_allow_html=True)
    #elif np.argmax(prediction) == 2:
    else:
        m3 = '<p style="color:Blue; font-size: 30px;">No Mask</p>'
        st.markdown(m3, unsafe_allow_html=True)
        #st.subheader("None of these")


    #st.text("Probability (0: m1, 1: m2, 2: m3)")
    prob = '<p style=" color:Black; font-size: 20px;">Probability </p>'  # (0: m1, 1: m2, 2: m3)  #font-family:sans-serif;
    st.markdown(prob, unsafe_allow_html=True)


    st.write(("Wearing Mask Incorrectly:  "),np.round(prediction[0][0], 7))
    st.write(("Wearing a Mask :   "),np.round(prediction[0][1], 7))
    st.write(("No Mask :   "),np.round(prediction[0][2], 7))