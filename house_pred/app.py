
import numpy as np

import pickle
model = pickle.load(open("./model.pkl", "rb"))

import streamlit as st
st.title("House Price Predictor using Linear Regression")

plotsize = st.text_input(label="Enter Plot Size")
bed = st.text_input(label="Enter Bedroom Count")
bath = st.text_input(label="Enter Bathroom Count")
store = st.text_input(label="Enter Store Count")

def check(string: str):
    if string == "yes":
        return 1
    return 0

drive_away = check(st.selectbox(label="Select Drive Away", options=["yes","no"]))
recroom = check(st.selectbox(label="Select Recroom", options=["yes","no"]))
fullbase = check(st.selectbox(label="Select fullbase", options=["yes","no"]))
gashw = check(st.selectbox(label="Select gashw", options=["yes","no"]))
airco = check(st.selectbox(label="Select airco", options=["yes","no"]))
garage = check(st.selectbox(label="Select garage", options=["yes","no"]))
area = check(st.selectbox(label="Select area", options=["yes","no"]))


if st.button(label="Predict"):
    num = model.predict(np.array([int(plotsize), int(bed), int(bath), int(store), drive_away, recroom, fullbase, gashw, airco, garage, area]).reshape(1,11))[0]
    st.text(num) 