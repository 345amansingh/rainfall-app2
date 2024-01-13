import streamlit as st
import pickle
import numpy as np

# Load the trained model
model_file = 'model.pkl'
model = pickle.load(open(model_file, 'rb'))

# Function to predict weather
def predict_weather(features):
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    return prediction

# Streamlit App
def main():
    st.title('Weather Prediction App')

    # User input for features
    precipitation = st.text_input('Enter Precipitation (in inches):', '1.0')
    temp_max = st.text_input('Enter Max Temperature (in Celsius):', '20.0')
    temp_min = st.text_input('Enter Min Temperature (in Celsius):', '10.0')
    wind = st.text_input('Enter Wind Speed (in mph):', '5.0')

    # Make a prediction
    if st.button('Predict Weather'):
        input_features = [float(precipitation), float(temp_max), float(temp_min), float(wind)]
        prediction = predict_weather(input_features)

        # Display the predicted weather
        st.subheader('Predicted Weather:')
        if prediction == 0:
            st.write('Drizzle')
        elif prediction == 1:
            st.write('Fog')
        elif prediction == 2:
            st.write('Rain')
        elif prediction == 3:
            st.write('Snow')
        else:
            st.write('Sun')

if __name__ == '__main__':
    main()
