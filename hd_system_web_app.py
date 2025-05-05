import numpy as np
import pickle
import streamlit as st
import pandas as pd
import requests
url = 'https://github.com/Kanda-com/Heart_D_Model/blob/main/naive_model_trained%20(1).sav'
loaded_model = requests.get(url)

with open('trained_model1.sav', 'wb') as f:
    pickle.dump(loaded_model, f)
with open('trained_model1.sav', 'rb') as f:
    loaded_model = pickle.load(f)
    
def hearth_disease_prediction(input_data):
    input_data_as_numpy_array=np.array(input_data)
    input_data.reshaped= input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data.reshaped)

    if prediction [0] == 0:
        return "The Person Does not have a Heart Disease"
    else:
        return "The person has heart disease"
def main():
    st.title("Heart Disease Prediction Machine Learning Model")
    age = st.text_imput("Enter the patient's Age 15-80 ")
    sex = st.text_input("Enter the patient's Gender (0[F],1[M])")
    Chest_Pain = st.text_input("Chest Pain Level(1,2 or 3)")
    Blood_Pressure = st.text_input("The Blood Pressure(mm hg)level (94-200)")
    cholestoral = st.text_input("Cholestoral level(mg/dl) (131 - 290)")
    
    age = pd.to_numeric(age, errors='coerce')
    sex = pd.to_numeric(sex, errors='coerce')
    Chest_Pain = pd.to_numeric(Chest_Pain, errors='coerce')
    Blood_Pressure =pd.to_numeric(Blood_Pressure, errors='coerce')
    cholestoral =pd.to_numeric(cholestoral, errors='coerce')
    
    diagnosis = ''
    
    if  st.button("PREDICT"):
        diagnosis = hearth_disease_prediction([age,sex,Chest_Pain,Blood_Pressure,cholestoral])
        st.success(diagnosis)

    if __name__ == '__main__':
        main()