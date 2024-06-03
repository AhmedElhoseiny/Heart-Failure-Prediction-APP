import pickle
import pandas as pd
import numpy as np
import streamlit as st

with open('model.pkl', 'rb') as f:
    Model = pickle.load(f)
with open('Inputs.pkl', 'rb') as f:
    Inputs = pickle.load(f)
    
def prediction(Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope):
    df = pd.DataFrame(columns=Inputs)
    df.at[0,"Age"] = Age    
    df.at[0,"ChestPainType"] = ChestPainType
    df.at[0,"RestingBP"] = RestingBP
    df.at[0,"Cholesterol"] = Cholesterol
    df.at[0,"RestingECG"] = RestingECG
    df.at[0,"MaxHR"] = MaxHR
    df.at[0,"Oldpeak"] = Oldpeak
    df.at[0,"ST_Slope"] = ST_Slope
    
    if FastingBS == "Yes" :
        df.at[0,"FastingBS"] = 1  
    else:
        df.at[0,"FastingBS"] = 0
        
    if Sex == "Male" :
        df.at[0,"Sex"] = "M"  
    else:
        df.at[0,"Sex"] = "F"
    
    if ExerciseAngina == "Yes" :
        df.at[0,"ExerciseAngina"] = "Y"  
    else:
        df.at[0,"ExerciseAngina"] = "N"
    
    df_encodded = pd.get_dummies(data=df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'] ,drop_first="True")*1

    for col in df_encodded.select_dtypes(include=['object']).columns:
        df_encodded[col] = pd.to_numeric(df_encodded[col], errors='coerce')
        
    result = Model.predict(df_encodded)[0]
    
    return result

def Main():
    st.title("Heart Failure Prediction")
    Sex = st.selectbox("Gender",['Male', 'Female'])
    ChestPainType = st.selectbox("Chest Pain Type",['ASY', 'NAP', 'ATA', 'TA'])
    Age = st.slider("Age",min_value=15.0 , max_value=100.0 , step=1.0,value = 1.0)
    RestingBP = st.slider("Resting Blood Presure",min_value=0.0 , max_value= 200.0 , step=1.0,value = 1.0)
    Cholesterol = st.slider("Cholesterol Level",min_value=0.0 , max_value=600.0 , step=1.0,value = 1.0)
    FastingBS = st.selectbox("Fasting Blood Suger",['Yes','No'])
    RestingECG = st.selectbox("Resting Rlectrocardiogram Result",['Normal', 'LVH','ST'])
    MaxHR = st.slider("MaxHR",min_value=60.0 , max_value=202.0 , step=1.0,value = 1.0)
    ExerciseAngina = st.selectbox("Exercise Angina",['Yes','No'])
    Oldpeak = st.slider("Old Peak",min_value=-2.6 , max_value=6.2 , step=1.0,value = 1.0)
    ST_Slope = st.selectbox("ST Slope",['Flat','Up','Down'])
    
    linkedin_url = "https://www.linkedin.com/in/ahmed-elhoseiny-2a952122a"
    github_url = "https://github.com/AhmedElhoseiny"
    email = "ahmedelhoseiny20022010@gmail.com"
    
    # Sidebar with contact information
    st.sidebar.image("Ahmed.jpg", width=100)
    st.sidebar.write("Connect with me:")
    st.sidebar.markdown(f"[![Email](https://img.shields.io/badge/Email-Contact-informational)](mailto:{email})")
    st.sidebar.markdown(f"[![GitHub](https://img.shields.io/badge/GitHub-Profile-green)]({github_url})")
    st.sidebar.markdown(f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue)]({linkedin_url})")

    if st.button("Predict"):
        result = prediction(Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope)
        list_result = ["No Heart Disease" , "HeartDisease"]
        st.text(list_result[result])
Main()
