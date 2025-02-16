import streamlit as st 
import pandas as pd 
import numpy as np 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

st.header(" :blue[Estimated Time Arrival Predictor] :car:")  
st.subheader("This tool helps to predict the estimated time arrival based on the weather,hour,traffic and week conditions from the starting point to the destination based upon longitude and latitude")
df = pd.read_csv("hyderabad_eta_data.csv")

# Sliders in the middle of the page
start_lat = st.slider(
    "Select Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=0.0,
    step=0.1
)

start_lon = st.slider(
    "Select start Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=0.0,
    step=0.1
)


end_lat = st.slider(
    "Select end  Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=0.0,
    step=0.1
)


end_lon = st.slider(
    "Select end Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=0.0,
    step=0.1
)





distance = st.slider(
    "Select the distance (km)",
    min_value = 0.0,
    max_value = 700.0,
    value = 100.0,
    step =1.0
)


traffic = st.selectbox("Select the traffic density", [1,  2, 3, 4, 5, 6, 7, 8, 9, 10])


weather = st.selectbox("Select the weather condition", ["rainy","clear","foggy"])

week_day = st.selectbox("Select the week",["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"])
week_val = {"Sunday" : 0,"Monday" : 1,"Tuesday" : 2,"Wednesday": 3,"Thursday": 4,"Friday": 5,"Saturday" : 6}
week = week_val[week_day]



hour = st.slider(
    "select the hour",
    min_value = 0,
    max_value = 23,
    value = 0,
    step = 1
)


# Map user-selected weather condition
weather_map = {'rainy': 0, 'clear': 1, 'foggy': 2}
weather_val = weather_map[weather]

# Prepare dataset
df["weather_condition"] = df["weather_condition"].map(weather_map)
X = df.drop("ETA", axis=1)
y = df["ETA"]

# splitting the target variable 
X = df.drop("ETA", axis =1)
y = df["ETA"]

# doing the train-test split
X_train, X_test,y_train,y_test = train_test_split(X,y , test_size= 0.2, random_state=23)

# Model Building and Model Evaluation 
sv_r = SVR(kernel='rbf', C=100, epsilon=0.1)
sv_r.fit(X_train,y_train)

y_pred = sv_r.predict(X_test)
result = mean_absolute_error(y_pred,y_test)

features = np.array([[start_lat, start_lon,end_lat,end_lon,distance,weather_val,traffic,week,hour]])

estimated_arrival = sv_r.predict(features)[0]

# Display formatted output without array brackets
if st.button(" :red[Click to predict estimated time]"):
    st.header(" :orange[The ESTIMATED TIME ARRIVAL IS ]")
    st.markdown(f"<h2 style='color:brown; font-size:36px;'><strong>{estimated_arrival:.2f} hours</strong></h2>", unsafe_allow_html=True)
