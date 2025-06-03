import streamlit as st
from PIL import Image
import skimage
from preprocessing import *
from model import Model 
if __name__ == '__main__':
    svc_1 = Model()
    svc_1.train()
    st.title('Cat Dog Classification')
    upload_file=st.file_uploader(
        'Input Image',type=["jpg", "jpeg", "png"]
    )
    if upload_file:
        print(type(upload_file.read()))
        fd,hog_image_rescaled = preprocessing(upload_file)
        col1,col2 = st.columns(2)
        col1.image(
            upload_file,caption='Your Image'
        )
        col2.image(hog_image_rescaled,caption='Hog Image')
        if st.button('Predict'):
            class_name,score=svc_1.infer(fd)
            col1,col2 = st.columns(2)
            col1.text(f'Class: {class_name}')
            col2.text(f'Score: {score}')
    # if f:



