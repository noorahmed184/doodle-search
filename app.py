import json
import numpy as np
import streamlit as st
from PIL import Image
from search_core import load_model, preprocess_query, embed, search
 
@st.cache_resource
def get_model():
    return load_model()
 
 
@st.cache_resource
def get_index():
    embeddings = np.load("index/embeddings.npy")
    images = np.load("index/gallery_images.npy")
    config = json.load(open("index/config.json"))
    return embeddings, images, config
 
 
model = get_model()
embeddings, gallery_images, config = get_index()
 
st.title("DoodleDex")
 
image = None
tab1, tab2 = st.tabs(["Upload", "Camera"])
with tab1:
    file = st.file_uploader("Upload a drawing", type=["png", "jpg", "jpeg"])
    if file is not None:
        image = Image.open(file)
with tab2:
    photo = st.camera_input("Photograph a drawing on paper (not people!)")
    if photo is not None:
        image = Image.open(photo)
 

if image is not None:
    small = preprocess_query(image)                  
 
    if small is None:
        st.info("No drawing found in the image")
    else:
        col1, col2 = st.columns(2)
        col1.image(image, caption="Your image", width=200)
        col2.image(Image.fromarray(small).resize((200, 200), Image.NEAREST),
                   caption="What the model sees")
 
        vector = embed(model, small[None])[0]       
        best, scores = search(vector, embeddings, config["top_k"])
 
        if scores[0] < config["threshold"]:
            st.warning("Nothing similar found")
        else:
            st.subheader("Top 25 similar doodles")
            for row in range(5):
                columns = st.columns(5)
                for c in range(5):
                    n = row * 5 + c
                    doodle = Image.fromarray(gallery_images[best[n]]).resize((100, 100), Image.NEAREST)
                    columns[c].image(doodle, caption=f"{scores[n]:.2f}")
 
st.caption("Data: Quick, Draw! by Google, Inc., CC BY 4.0")
