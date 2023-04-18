import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Pnuemonia Detection Image Classifier")
st.text("Provide URL of Chest Xray for Pneumonia Detection")


# @st.cache(allow_output_mutation=True)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('C:\\Users\\umang\\OneDrive\\Desktop\\Pneumonia detection\\efficientnetb0-model.h5')
    return model


with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes = ['Normal','Pneumonia']


def decode_img(image):
    img = tf.image.decode_jpeg(image, channels=3)
    img = tf.image.resize(img, [224, 224])
    return np.expand_dims(img, axis=0)


path = st.text_input('Enter Image URL to Classify.. ',
                     'https://raw.githubusercontent.com/akashprasad7631/ML-Lung_infection_detection/main/val/PNEUMONIA/person1946_bacteria_4875.jpeg')
if path is not None:
    response = requests.get(path)
    # get the content of the image as bytes
    content = response.content
    st.image(content, caption='INPUT')
    # content = requests.get(path).content

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
        label = np.argmax(model.predict(decode_img(content)), axis=1)
    print(model.predict(decode_img(content)))
    print(label)
    st.write(classes[label[0]])
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Pneumonia Detection', use_column_width=True)
