import streamlit as st
import pandas as pd
import joblib

# Model load
model = joblib.load("pass_fail_predictor.pkl")

st.title("Student Pass/Fail Predictor")

marks = st.number_input(
    "Marks",
    min_value=0,
    max_value=100,
    value=50
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

if st.button("Predict"):

    data = pd.DataFrame(
        [[marks, attendance]],
        columns=["marks", "attendance"]
    )

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1] * 100

    if prediction == 1:
        st.success("PASS")
    else:
        st.error("FAIL")

    st.write(f"Chance of Passing: {probability:.2f}%")
