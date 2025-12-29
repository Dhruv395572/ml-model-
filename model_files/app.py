import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="MedAlert AI",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 MedAlert AI – Disease Prediction System")
st.write("Select symptoms and predict possible disease using Machine Learning.")

# -------------------------------
# Load model & encoder
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("disease_model.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, le

model, le = load_model()

# -------------------------------
# Load feature (symptom) names
# IMPORTANT: same order as training
# -------------------------------
@st.cache_data
def load_features():
    # load training file only to get column names
    df = pd.read_csv("archive/Training.csv")
    df.drop(columns=["Unnamed: 133"], inplace=True, errors="ignore")
    X = df.drop("prognosis", axis=1)
    return X.columns.tolist()

symptoms = load_features()

st.markdown("### 🤒 Select Symptoms")

# -------------------------------
# Symptom selection (checkboxes)
# -------------------------------
cols = st.columns(4)
selected = []

for i, symptom in enumerate(symptoms):
    with cols[i % 4]:
        if st.checkbox(symptom.replace("_", " ").title()):
            selected.append(symptom)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Disease"):
    if len(selected) == 0:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # create input vector
        input_data = pd.DataFrame(
            [[0]*len(symptoms)],
            columns=symptoms
        )

        for s in selected:
            input_data[s] = 1

        # predict
        pred_class = model.predict(input_data)
        pred_disease = le.inverse_transform(pred_class)[0]

        st.success(f"🧠 Predicted Disease: **{pred_disease}**")

        st.info(
            "⚠️ This is an AI-based prediction system and not a medical diagnosis. "
            "Please consult a doctor for professional advice."
        )

# -------------------------------
# Sidebar info
# -------------------------------
st.sidebar.title("ℹ️ About MedAlert AI")
st.sidebar.write(
    """
    **MedAlert AI** is a machine learning based disease prediction system.
    
    - Trained using Random Forest
    - Uses symptom-based prediction
    - Supports multiple diseases
    """
)

st.sidebar.markdown("### 🧬 Supported Diseases")
st.sidebar.write(list(le.classes_))
