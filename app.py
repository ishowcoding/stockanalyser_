import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import tensorflow as tf
import yfinance as yf
from keras.models import load_model
import datetime

st.title('Stock Analysis')
start_date = st.date_input('Enter Start Date', value=datetime.date(2012, 1, 1))
end_date = st.date_input('Enter End Date', value=datetime.date(2022, 1, 1))

if start_date < end_date:
    st.success('Start date: `{}`\n\nEnd date:`{}`'.format(start_date, end_date))
else:
    st.error('Error: End date must fall after start date.')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start_date, end=end_date)

st.subheader('Data from {} - {}'.format(start_date, end_date))
st.write(df.describe())



st.subheader( 'Closing Price vs Time chart')
fig = plt.figure(figsize =(12,6))
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
m100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(m100)
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA and 200MA')
m100 = df.Close.rolling(100).mean()
m200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(m100)
plt.plot(m200)
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training) 

x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i: 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

model = load_model('keras_model.h5')


past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(data_training_array[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = -100*y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader( 'Predicted Stock Price')
fig=plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b',label = 'Orginal Price')
plt.plot(y_predicted, 'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.write("<div style='text-align: center; margin-top: 70px; font-size: 18px;'>Build by | Sagar Sugunan | Sajjad Saheer Ali | Nishan P | ",unsafe_allow_html=True)
