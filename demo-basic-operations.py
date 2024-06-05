

import streamlit as st
st.title("看什么看")
st.header("给你看看水果之王")
st.subheader("包好吃的")
st.text("包香的")

st.markdown("this is 榴莲 \n \
            ![](https://img2.baidu.com/it/u=3515280952,3740439345&fm=253&fmt=auto&app=120&f=JPEG?w=749&h=500)")

if st.checkbox("Show/Hide"):
    st.text("榴莲包好吃的吧")


status = st.radio("select gender:" ,
                  ('香',
                   '香死了'))
if status == '香':
    st.success("小伙子有前途")
else:
    st.success("无敌了你")

hobbies = st.multiselect("Hobbies:",
               ['吃',
                '吃吃',
                '吃吃吃',
                '我库库吃'])
st.write("You selected: ", hobbies)

if st.button("about"):
    st.text("你不吃你真没品")

name = st.text_input("Enter your name:")
if st.button("Submit"):
    st.write("Hello, ", name)

level = st.slider("Select your level", 1, 5)
st.write("You selected: ", level)

from fastai.vision.all import *
upload_img = st.file_uploader("Upload an image",
                               type=['jpg',
                                      'png'])

if upload_img is not None:
    img = PILImage.create(upload_img)
    st.image(img.to_thumb(256,256), 
             caption="Uploaded Image")