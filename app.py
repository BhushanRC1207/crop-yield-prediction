import streamlit as st
import joblib
import numpy as np
import pandas as pd


model = joblib.load('dtr.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.set_page_config(
    page_title = 'Crop Yield Prediction',
    layout = 'wide'
)

st.title('Crop Yield Prediction')

train = pd.read_csv('train.csv')


year = st.number_input('Year',
                       min_value=1990,
                       max_value=2024,
                       step=1)

area = st.selectbox(
	"Area",
	options=train.Area.unique()
)

item = st.selectbox(
	"Item",
	options=train.Item.unique()
)


average_rain_fall_mm_per_year = st.number_input('Average Rainfall (in mm)',
                                                min_value=0.0,
                                                max_value=5000.0,
                                                step=0.1)

pesticides_tonnes = st.number_input('Pesticides Tonnes',
                                    min_value=0.0,
                                    max_value=367778.0,
                                    step=0.1)

avg_temp = st.number_input('Average Temperature',
                           min_value=-10.0,
                           max_value=50.0,
                           step=0.1)




if st.button('Predict'):

    features = np.array([[year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,area,item]],dtype=object)
    transformed_features = preprocessor.transform(features)
    prediction = model.predict(transformed_features).reshape(1,-1)

    st.markdown(f"""
    <div>
        <p style="font-size: 20px; font-weight: bold; color: white;">Predicted Crop Yield : {int(prediction)}</p>
    </div>
    """, unsafe_allow_html=True)