import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# Streamlit App
st.set_page_config(page_title="Diabetes Prediction App", page_icon="⚕️", layout="wide")
# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("pima-data.csv")
    return df

data = load_data()

data["diabetes"] = data["diabetes"].map({True: 1, False: 0})

# Define features and labels
feature_columns = ["num_preg", "glucose_conc", "diastolic_bp", "insulin", "bmi", "diab_pred", "age", "skin"]
X = data[feature_columns]
y = data["diabetes"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model
with open("diabetes_model.pkl", "wb") as file:
    pickle.dump((model, scaler), file)


st.title("🩺 Diabetes Prediction App")
st.markdown("### Enter patient details to predict diabetes")

# User Input Fields with Animations
st.sidebar.header("Enter Patient Data 🏥")

def user_input_features():
    num_preg = st.sidebar.number_input("👶 Number of Pregnancies", 0, 20, 1)
    glucose_conc = st.sidebar.number_input("🩸 Glucose Concentration", 0, 200, 100)
    diastolic_bp = st.sidebar.number_input("💓 Diastolic Blood Pressure", 0, 150, 70)
    insulin = st.sidebar.number_input("💉 Insulin Level", 0, 900, 30)
    bmi = st.sidebar.number_input("⚖️ BMI", 0.0, 70.0, 25.0)
    diab_pred = st.sidebar.number_input("🧬 Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.number_input("🎂 Age", 1, 120, 30)
    skin = st.sidebar.number_input("🦵 Skin Thickness", 0.0, 2.0, 1.0)

    
    data = np.array([[num_preg, glucose_conc, diastolic_bp, insulin, bmi, diab_pred, age, skin]])
    return data

# Get user input
data = user_input_features()

# Load model and scaler
with open("diabetes_model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

# Predict diabetes with Loading Animation
if st.button("🔍 Predict Diabetes"):  
    with st.spinner("Analyzing data... 🧪"):
        time.sleep(2)  # Simulate processing time
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    
    if result == "Non-Diabetic":
        st.success(f"### ✅ Prediction: {result}")
    else:
        st.error(f"### ⚠️ Prediction: {result}")    
    
    if result == "Diabetic":
        st.warning("⚠️ You might have diabetes. Please consult a doctor immediately.")
    else:
        st.success("🎉 No signs of diabetes detected. Keep maintaining a healthy lifestyle!")

# Visualization
st.subheader("📊 Data Overview")
st.dataframe(data)

df = load_data()
fig = px.histogram(df, x="glucose_conc", color="diabetes", barmode="overlay", title="Glucose Levels Distribution")
st.plotly_chart(fig)
