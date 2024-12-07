import streamlit as st
from fastai.vision.all import *

def is_cat(file_name):
    return file_name[0].isupper()

cat_vs_dog_model = load_learner("cat_vs_dog_model.pkl")

def predict(file_name):
    img = PILImage.create(file_name)
    prediction, idx, accuracy = cat_vs_dog_model.predict(img)

    if prediction == "True":

        return "Cat"
    else:
        return f"Dog {accuracy}%"

st.text("Cat vs Dog Classifier")
st.text("Built by Jayden Hang")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    prediction = predict(uploaded_file)
    st.image(uploaded_file, caption=prediction, use_column_width=True)
    st.text(accuracy)

