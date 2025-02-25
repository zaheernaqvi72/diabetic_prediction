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

csv_data = load_data()

csv_data["diabetes"] = csv_data["diabetes"].map({True: 1, False: 0})

# Define features and labels
feature_columns = ["num_preg", "glucose_conc", "diastolic_bp", "insulin", "bmi", "diab_pred", "age", "skin"]
X = csv_data[feature_columns]
y = csv_data["diabetes"]

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

st.subheader("📊 Complete Dataset Overview")
st.dataframe(csv_data)

st.subheader("📝 User-Entered Data")
user_df = pd.DataFrame(data, columns=["num_preg", "glucose_conc", "diastolic_bp", "insulin", "bmi", "diab_pred", "age", "skin thickness"])
st.dataframe(user_df)

df = load_data()

# Histogram: Glucose Levels vs. Diabetes
fig1 = px.histogram(df, x="glucose_conc", color="diabetes", barmode="overlay", title="Glucose Levels Distribution")
st.plotly_chart(fig1)

# Histogram: BMI vs. Diabetes
fig2 = px.histogram(df, x="bmi", color="diabetes", barmode="overlay", title="BMI Distribution")
st.plotly_chart(fig2)

# Histogram: Age vs. Diabetes
fig3 = px.histogram(df, x="age", color="diabetes", barmode="overlay", title="Age Distribution")
st.plotly_chart(fig3)

# Box Plot: Blood Pressure vs. Diabetes
fig4 = px.box(df, x="diabetes", y="diastolic_bp", color="diabetes", title="Blood Pressure vs. Diabetes")
st.plotly_chart(fig4)

# Box Plot: Insulin Levels vs. Diabetes
fig5 = px.box(df, x="diabetes", y="insulin", color="diabetes", title="Insulin Levels vs. Diabetes")
st.plotly_chart(fig5)

# Scatter Plot: Glucose vs. BMI (Colored by Diabetes)
fig6 = px.scatter(df, x="glucose_conc", y="bmi", color="diabetes", title="Glucose vs. BMI (Colored by Diabetes)")
st.plotly_chart(fig6)

# Violin Plot: BMI Distribution by Diabetes Status
fig7 = px.violin(df, x="diabetes", y="bmi", color="diabetes", title="BMI Distribution by Diabetes Status", box=True)
st.plotly_chart(fig7)

# Violin Plot: Glucose Concentration Distribution
fig8 = px.violin(df, x="diabetes", y="glucose_conc", color="diabetes", title="Glucose Concentration Distribution", box=True)
st.plotly_chart(fig8)

# Pair Plot: Multiple Features Analysis (Scatter matrix)
st.subheader("Pairwise Feature Relationships")
selected_features = ["glucose_conc", "bmi", "age", "diastolic_bp", "insulin"]
fig9 = px.scatter_matrix(df, dimensions=selected_features, color="diabetes", title="Pair Plot of Key Features")
st.plotly_chart(fig9)

# Heatmap: Correlation Matrix
st.subheader("Feature Correlation Heatmap")
import seaborn as sns
import matplotlib.pyplot as plt

fig10, ax = plt.subplots()
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
st.pyplot(fig10)
