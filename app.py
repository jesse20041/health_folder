import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")


@st.cache_resource
def load_pipeline():
    return joblib.load("rf_pipeline.pkl")


pipeline = load_pipeline()
model = pipeline["model"]
scaler = pipeline["scaler"]
features = pipeline["features"]  # exact column order the model was trained on

# These 5 were the ones scaled in the notebook — everything else is categorical/left as-is
numerical_features = ["age", "resting_blood_pressure", "cholesterol", "max_heart_rate", "st_depression"]

st.title("❤️ Heart Disease Risk Predictor")
st.write(
    "Enter patient details below to estimate the likelihood of heart disease. "
    "This tool is for educational purposes only and is **not** a substitute "
    "for professional medical advice."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=54)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    chest_pain_type = st.selectbox(
        "Chest Pain Type", options=[
            ("Typical Angina", 0), ("Atypical Angina", 1),
            ("Non-anginal Pain", 2), ("Asymptomatic", 3),
        ], format_func=lambda x: x[0])[1]
    resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=246)
    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    ecg = st.selectbox(
        "Resting ECG Results", options=[
            ("Normal", 0), ("ST-T Wave Abnormality", 1), ("Left Ventricular Hypertrophy", 2),
        ], format_func=lambda x: x[0])[1]

with col2:
    max_heart_rate = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exercise_induced_chest_pain = st.selectbox("Exercise-Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    st_depression = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
    st_slope = st.selectbox(
        "Slope of Peak Exercise ST Segment", options=[
            ("Upsloping", 0), ("Flat", 1), ("Downsloping", 2),
        ], format_func=lambda x: x[0])[1]
    stained_blood_vessels = st.selectbox("Number of Major Vessels (0-3) Colored by Fluoroscopy", options=[0, 1, 2, 3])
    blood_disorder = st.selectbox(
        "Blood Disorder Code (as coded in your dataset)", options=[0, 1, 2, 3], index=2,
        help="Check data_01.xls to confirm what each code represents in your dataset (commonly: normal / fixed defect / reversible defect)."
    )

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    row = {
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_blood_pressure,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "ecg": ecg,
        "max_heart_rate": max_heart_rate,
        "exercise_induced_chest_pain": exercise_induced_chest_pain,
        "st_depression": st_depression,
        "st_slope": st_slope,
        "stained_blood_vessels": stained_blood_vessels,
        "blood_disorder": blood_disorder,
    }

    # Build the dataframe in the EXACT column order the model was trained on
    new_data = pd.DataFrame([row])[features]

    # Scale only the numerical features, using the scaler fitted during training
    new_data_scaled = new_data.copy()
    new_data_scaled[numerical_features] = scaler.transform(new_data[numerical_features])

    prediction = model.predict(new_data_scaled)[0]
    proba = model.predict_proba(new_data_scaled)[0]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ Higher risk of heart disease (confidence: {proba[1]*100:.1f}%)")
    else:
        st.success(f"✅ Lower risk of heart disease (confidence: {proba[0]*100:.1f}%)")

    with st.expander("See input summary"):
        st.dataframe(new_data)

st.caption("Model: RandomForestClassifier · For demonstration purposes only.")