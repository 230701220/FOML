import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter your health details below:")

# Input fields
preg = st.number_input("Pregnancies", 0)
glucose = st.number_input("Glucose", 0)
bp = st.number_input("Blood Pressure", 0)
skin = st.number_input("Skin Thickness", 0)
insulin = st.number_input("Insulin", 0)
bmi = st.number_input("BMI", 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Age", 0)

# Optional threshold slider
st.markdown("---")
threshold = st.slider("🔧 Decision Threshold (default: 0.5)", 0.0, 1.0, 0.5, step=0.01)

# Predict button
if st.button("🔍 Predict"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    data_scaled = scaler.transform(data)
    prob = model.predict_proba(data_scaled)[0][1]
    prediction = 1 if prob >= threshold else 0

    # Risk category based on probability
    if prob < 0.4:
        risk = "Low"
        color = "green"
    elif prob < 0.7:
        risk = "Moderate"
        color = "orange"
    else:
        risk = "High"
        color = "red"

    st.markdown("---")
    if prediction == 1:
        st.error(f"🚨 **Prediction: Diabetic**")
    else:
        st.success(f"✅ **Prediction: Not Diabetic**")

    st.markdown(f"**🧪 Probability:** `{prob:.2f}`")
    st.markdown(f"**⚠️ Risk Level:** :{color}[{risk}] (based on probability)")

# Optional: Feature importance plot
if hasattr(model, 'feature_importances_'):
    st.subheader("🔬 Feature Importances")
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance")

    fig, ax = plt.subplots()
    ax.barh(imp_df["Feature"], imp_df["Importance"], color='teal')
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importances")
    st.pyplot(fig)
