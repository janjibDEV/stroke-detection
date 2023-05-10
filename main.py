import streamlit as st
from streamlit_extras.mention import mention
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb

model = pickle.load(open("model.pkl","rb"))
# title
st.title("Stroke Prediction by janjibDEV")
# header
st.text("The aim of this project is to showcase janjibDEV's ability to use machine learning") 
st.text("knowledge in a practical way.")
# sub header
st.header("About")
st.text("This is a supervised machine learning project where it can predict the probability") 
st.text("of having stroke based on input given by users. The dataset used in this ") 
st.text("project can be accessed on Kaggle https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset")
mention(
    label="Dataset",
    url="https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset",
)
# load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv.xls")
st.subheader("Data")
st.dataframe(df)
# Input
st.subheader("Input")
st.write("Please fill in your data")
# Data collection
gender = st.selectbox("Choose your gender",("Male","Female","Other"))
age = st.slider("Age",1,100,20)
agl = st.slider("Average Glucose Level",1.0,300.0,70.0,0.1)
bmi = round(st.number_input("Enter your BMI",value=20.0),1)
st.write("`bmi = weight (kg) / height (m^2)`")
isHypertension = st.radio(
    "Have you been diagnosed with hypertension ?",
    ('Yes', 'No'),1)
isHeartDisease = st.radio(
    "Have you been diagnosed with heart disease ?",
    ('Yes', 'No'),1)
everMarried = st.radio(
    "Have you ever married ?",
    ('Yes', 'No'),0)
workType = st.radio(
    "Choose one that is relevant to your job ?",
    ('Private', 'Self-employed',"Government", "Not Working", "Children"),2)
residenceType = st.radio(
    "Choose one that is relevant to your home ?",
    ('Urban', 'Rural'),0)
smoking_status = st.radio(
    "Choose one that is relevant to your smoking status ?",
    ('Smokes', 'Never Smoke', 'Formerly smoked','Unknown'),0)

listAtt = [age]
# cleaner function
def noYesConverter(x):
    return bool(x == "Yes")

for i in [isHypertension,isHeartDisease,everMarried]:
    listAtt.append(noYesConverter(i))

# ['gender_Female', 'gender_Male', 'work_type_Govt_job',
#        'work_type_Never_worked', 'work_type_Private',
#        'work_type_Self-employed', 'work_type_children',
#        'Residence_type_Rural', 'Residence_type_Urban',
#        'smoking_status_Unknown', 'smoking_status_formerly smoked',
#        'smoking_status_never smoked', 'smoking_status_smokes']

def gender_cleaner(x):
    if x == "Male":
        return [0,1]
    else:
        return [1,0]
    
def work_type_cleaner(x):
    dc = {"Government":0,"Not Working":1,"Private":2,"Self-employed":3,"Children":4}
    lst = [0 for x in range(len(dc))]
    lst[dc[x]] = 1
    return lst

def residence_cleaner(x):
    if x == "Rural":
        return [1,0]
    else:
        return [0,1]
    
def smoking_cleaner(x):
    dc = {"Unknown":0,"Formerly smoked":1,"Never Smoke":2,"Smokes":3}
    lst = [0 for x in range(len(dc))]
    lst[dc[x]] = 1
    return lst

listAtt.extend([agl,bmi])
listAtt += gender_cleaner(gender)
listAtt += work_type_cleaner(workType)
listAtt += residence_cleaner(residenceType)
listAtt += smoking_cleaner(smoking_status)


if st.button('Submit'):
    
    prediction = model.predict([listAtt])
    if prediction[0] == 0:
        st.text("Congrats. The system predicts you don't have stroke")
    else:
        st.text("Ooops. The system predicts you do have stroke. Take care")

