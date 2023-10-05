import streamlit as st
import pandas as pd
import joblib
import os

# Überprüfung, ob die Daten überhaupt im Session State sind
if 'final_df' in st.session_state:
    df = st.session_state.final_df

# Ermitteln der aktuellen Directory
current_directory = os.path.dirname(__file__)

# Suchen nach dem Pipeline-File in der Parent-Directory
pipeline_file_path = os.path.join(os.path.dirname(current_directory), 'trained_pipeline.pkl')

# Überprüfen, ob das Pipeline-File da ist, sonst Fehlermeldung
if os.path.exists(pipeline_file_path):
    loaded_pipe = joblib.load(pipeline_file_path)
else:
    st.error(f'Trained pipeline file not found at: {pipeline_file_path}')
    st.stop()


# Select der App
make = st.selectbox('Car Make', df['make'].unique(), key='make')
model = st.selectbox('Car Model', df['model'].unique(), key='model')
fuel = st.selectbox('Fuel Type', df['fuel'].unique(), key='fuel')
gear = st.selectbox('Transmission Type', df['gear'].unique(), key='gear')
offer_type = st.selectbox('Offer Type', df['offerType'].unique(), key='offer_type')

# Inputfelder der App
year = st.number_input('Year', min_value=int(df['year'].min()), max_value=int(df['year'].max()), key='year')
mileage = st.number_input('Mileage (in kilometers)', min_value=int(df['mileage'].min()), max_value=int(df['mileage'].max()), key='mileage')

# Abwicklung des Berechnugnsprozess der Preises mit Hilfe eines Buttons
if st.button('Predict Price'):
    # Umwandlung der Daten in Dataframe
    input_data = pd.DataFrame({
        'make': [make],
        'model': [model],
        'fuel': [fuel],
        'gear': [gear],
        'offerType': [offer_type],
        'year': [year],
        'mileage': [mileage]
    })

    #Prediction
    predicted_price = loaded_pipe.predict(input_data)[0]

    st.write(f'Predicted Price: {predicted_price:.2f}€')