import pandas as pd
cars = pd.read_csv('./cleaned_cars.csv')

mark_list = list(set(cars['mark']))
model_list = list(set(cars['model']))
year_list = list(set(cars['year']))
fuel_type_list = list(set(cars['fuel']))
city_list = list(set(cars['city']))

# result = model.predict(pd.DataFrame([['nissan','patrol',2000,35000,2953,'Diesel','Rybarzowice']],columns = ['mark','model','year','mileage','vol_engine','fuel','city']))
# print(result)

import streamlit as st

st.title("Predict Car Price")

# Selector
# mark = st.selectbox(label="Select Car", options=mark_list)
# model = st.selectbox(label="Select Model", options=model_list)
# fuel_type = st.selectbox(label="Select Fuel Type", options=fuel_type_list)
# city = st.selectbox(label="Select City", options=city_list)

year = st.select_slider(label="Select Year", options=year_list)

# Numeric Input
mileage = st.text_input(label="Enter Mileage")
vol_engine = st.text_input(label="Enter Engine Volume")

import pickle
model = pickle.load(open("LinearRegression2.pkl",'rb'))

if st.button(label="Predict"):
    result = model.predict(pd.DataFrame([[int(year),int(mileage),int(vol_engine)]],columns = ['year','mileage','vol_engine']))[0]
    print(result)
    st.text(result)