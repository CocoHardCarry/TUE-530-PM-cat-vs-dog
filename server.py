import streamlit as st
from fastai.vision.all import *

def is_cat(file_name):
    return file_name[0].isupper()

cat_vs_dog_modl = load_learner("cat_vs_dog_model.pkl")

st.text("Cat vs Dog Classifier")
st.text("Built by Jayden Hang")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)