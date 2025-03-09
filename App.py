import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load the pre-trained model (no scaler needed anymore)
model = joblib.load('weather_temp_model.pkl')

# Title of the app
st.title('Weather Forecast Dashboard')

# Description of the app
st.write("""
This dashboard allows you to predict the maximum temperature based on weather-related features.
Input the values for each feature, and the model will predict the maximum temperature for you.
""")

# Input fields for user to enter data
pressure = st.number_input('Pressure (hPa)', min_value=900.0, max_value=1050.0, value=1015.9)
temperature = st.number_input('Temperature (°C)', min_value=-10.0, max_value=50.0, value=21.3)
mintemp = st.number_input('Minimum Temperature (°C)', min_value=-10.0, max_value=50.0, value=20.7)
dewpoint = st.number_input('Dewpoint (°C)', min_value=-10.0, max_value=50.0, value=20.2)
humidity = st.number_input('Humidity (%)', min_value=0, max_value=100, value=95)
rainfall = st.selectbox('Rainfall', ['yes', 'no'])

# Convert 'rainfall' from 'yes'/'no' to 1/0
rainfall = 1 if rainfall == 'yes' else 0

# Sunshine input
sunshine = st.number_input('Sunshine (hours)', min_value=0, max_value=24, value=0)

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'pressure': [pressure],
    'temperature': [temperature],
    'mintemp': [mintemp],
    'dewpoint': [dewpoint],
    'humidity': [humidity],
    'rainfall': [rainfall],
    'sunshine': [sunshine]
})

# Ensure no spaces in column names by stripping them
input_data.columns = input_data.columns.str.replace(' ', '')

# Ensure the input data matches the expected format by the model
expected_columns = ['pressure', 'temperature', 'mintemp', 'dewpoint', 'humidity', 'rainfall', 'sunshine']

# Check if all expected columns are present
if all(col in input_data.columns for col in expected_columns):
    # Predict the maximum temperature
    try:
        predicted_temp = model.predict(input_data)
        st.write(f"**Predicted Maximum Temperature: {predicted_temp[0]:.2f}°C**")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
else:
    st.error("Input data does not match the expected feature set. Please check the input fields.")

# Visualization Section
st.subheader("Historical Data and Correlations")
st.write("Below is the correlation matrix and some historical data used to train the model:")

# Load your dataset for visualization (example dataset)
data = pd.read_csv('weather_data.csv')

# Show sample data
st.dataframe(data.head())

# Display Correlation Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot()  # Display the plot in the Streamlit app
