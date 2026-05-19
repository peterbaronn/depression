import streamlit as st
import pandas as pd
import joblib

# =========================================================
# LOAD MODEL
# =========================================================
model_package = joblib.load("depression_model.pkl")

model = model_package["model"]
scaler = model_package["scaler"]
feature_columns = model_package["feature_columns"]

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Student Depression Prediction",
    page_icon="🧠",
    layout="centered"
)

# =========================================================
# TITLE
# =========================================================
st.title("🧠 Student Depression Prediction")

st.write(
    """
    This application predicts student depression risk using
    **Tuned Gradient Boosting with SMOTE**.
    """
)

st.warning(
    """
    Disclaimer:
    This prediction is for educational purposes only
    and should not be used as a medical diagnosis.
    """
)

# =========================================================
# INPUT SECTION
# =========================================================
st.subheader("Input Student Data")

# Gender
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

# Age
age = st.slider(
    "Age",
    18,
    60,
    25
)

# Academic Pressure
academic_pressure = st.slider(
    "Academic Pressure",
    0.0,
    5.0,
    3.0,
    step=1.0
)

# GPA INPUT (0-4 SCALE)
cgpa_input = st.slider(
    "GPA / IPK",
    0.0,
    4.0,
    3.0,
    step=0.01
)

# CONVERT GPA 4 SCALE -> 10 SCALE
cgpa = cgpa_input * 2.5

# Study Satisfaction
study_satisfaction = st.slider(
    "Study Satisfaction",
    0.0,
    5.0,
    3.0,
    step=1.0
)

# Sleep Duration
sleep_duration = st.selectbox(
    "Sleep Duration",
    [
        "Less than 5 hours",
        "5-6 hours",
        "7-8 hours",
        "More than 8 hours",
        "Others"
    ]
)

# Dietary Habits
dietary_habits = st.selectbox(
    "Dietary Habits",
    [
        "Healthy",
        "Moderate",
        "Unhealthy",
        "Others"
    ]
)

# Suicidal Thoughts
suicidal_thoughts = st.selectbox(
    "Have you ever had suicidal thoughts?",
    [
        "No",
        "Yes"
    ]
)

# Work Study Hours
work_study_hours = st.slider(
    "Work/Study Hours",
    0.0,
    12.0,
    6.0,
    step=1.0
)

# Financial Stress
financial_stress = st.slider(
    "Financial Stress",
    1.0,
    5.0,
    3.0,
    step=1.0
)

# Family History
family_history = st.selectbox(
    "Family History of Mental Illness",
    [
        "No",
        "Yes"
    ]
)

# =========================================================
# PREPROCESS INPUT
# =========================================================
def preprocess_input():

    input_data = pd.DataFrame({

        "Gender": [gender],

        "Age": [age],

        "Academic Pressure": [academic_pressure],

        "CGPA": [cgpa],

        "Study Satisfaction": [study_satisfaction],

        "Sleep Duration": [sleep_duration],

        "Dietary Habits": [dietary_habits],

        "Have you ever had suicidal thoughts ?": [
            suicidal_thoughts
        ],

        "Work/Study Hours": [work_study_hours],

        "Financial Stress": [financial_stress],

        "Family History of Mental Illness": [
            family_history
        ]
    })

    # =====================================================
    # ENCODING
    # =====================================================

    # Gender
    input_data["Gender"] = input_data["Gender"].map({
        "Female": 0,
        "Male": 1
    })

    # Sleep Duration
    input_data["Sleep Duration"] = input_data[
        "Sleep Duration"
    ].map({
        "Others": 0,
        "Less than 5 hours": 1,
        "5-6 hours": 2,
        "7-8 hours": 3,
        "More than 8 hours": 4
    })

    # Dietary Habits
    input_data["Dietary Habits"] = input_data[
        "Dietary Habits"
    ].map({
        "Healthy": 0,
        "Moderate": 1,
        "Unhealthy": 2,
        "Others": 3
    })

    # Suicidal Thoughts
    input_data["Have you ever had suicidal thoughts ?"] = (
        input_data[
            "Have you ever had suicidal thoughts ?"
        ].map({
            "No": 0,
            "Yes": 1
        })
    )

    # Family History
    input_data["Family History of Mental Illness"] = (
        input_data[
            "Family History of Mental Illness"
        ].map({
            "No": 0,
            "Yes": 1
        })
    )

    # =====================================================
    # COLUMN ORDER
    # =====================================================
    input_data = input_data[feature_columns]

    # =====================================================
    # SCALING
    # =====================================================
    input_scaled = scaler.transform(input_data)

    return input_scaled

# =========================================================
# PREDICTION
# =========================================================
if st.button("Predict Depression Risk"):

    processed_input = preprocess_input()

    prediction = model.predict(processed_input)[0]

    probability = model.predict_proba(
        processed_input
    )[0][1]

    # =====================================================
    # RESULT
    # =====================================================
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            "Result: Depressed Risk Detected"
        )
    else:
        st.success(
            "Result: Not Depressed"
        )

    st.write(
        f"Depression Probability: "
        f"**{probability:.2%}**"
    )

    # =====================================================
    # RISK LEVEL
    # =====================================================
    if probability >= 0.70:

        st.warning(
            "Risk Level: High"
        )

    elif probability >= 0.40:

        st.info(
            "Risk Level: Moderate"
        )

    else:

        st.success(
            "Risk Level: Low"
        )
