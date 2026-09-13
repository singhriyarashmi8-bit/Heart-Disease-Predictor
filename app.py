import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Cardiovascular Risk Predictor",
    page_icon="🫀",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94A3B8;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #E11D48;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 8px;
        padding: 0.6rem;
        border: none;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #BE123C;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Models aur files load karna (cached for speed)
@st.cache_resource
def load_artifacts():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("Columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_artifacts()

# Header Section
st.markdown('<div class="main-title">🫀 Heart Disease Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter clinical parameters to evaluate cardiovascular risk using machine learning.</div>', unsafe_allow_html=True)

# 2. User Inputs (Organized into 2 clean columns)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📋 Patient Demographics & Vitals")
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ['M', 'F'])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], 
                              help="ATA: Atypical Angina, NAP: Non-Anginal, TA: Typical Angina, ASY: Asymptomatic")
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                              format_func=lambda x: "Yes (> 120 mg/dL)" if x == 1 else "No (≤ 120 mg/dL)")

with col2:
    st.markdown("### 🩺 Clinical & Exercise Tests")
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"],
                                  format_func=lambda x: "Yes" if x == "Y" else "No")
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.write("")

# 3. Prediction Button & Output Logic
if st.button("Run Risk Prediction"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    confidence = probabilities[prediction] * 100

    st.markdown("---")
    st.markdown("### 🔍 Assessment Output")
    
    res_col1, res_col2 = st.columns([2.5, 1])

    with res_col1:
        if prediction == 1:
            st.error("⚠️ **High Risk of Heart Disease**\n\nThe indicators suggest clinical patterns associated with cardiovascular risk.")
        else:
            st.success("✅ **Low Risk of Heart Disease**\n\nThe indicators currently fall within normal expected parameters.")

    with res_col2:
        st.metric(label="Prediction Confidence", value=f"{confidence:.1f}%")

st.markdown("---")
st.caption("⚠️ *Educational Disclaimer: This tool is built using machine learning benchmarks for educational demonstration and cannot replace certified medical diagnostics.*")
