
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("titanic_survival_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Title
st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter passenger information below and the machine learning model "
    "will predict the survival outcome."
)

# Passenger inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=25,
    step=1
)

sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=30.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)

# Prediction button
if st.button("Predict Survival"):

    # Create input DataFrame
    passenger = pd.DataFrame({
        "pclass": [pclass],
        "sex": [sex],
        "age": [age],
        "sibsp": [sibsp],
        "parch": [parch],
        "fare": [fare],
        "embarked": [embarked]
    })

    # Make prediction
    prediction = model.predict(passenger)[0]

    # Get probabilities
    probability = model.predict_proba(passenger)[0]

    survival_probability = probability[1] * 100
    death_probability = probability[0] * 100

    # Display result
    if prediction == 1:
        st.success("Prediction: Passenger is predicted to SURVIVE.")
    else:
        st.error("Prediction: Passenger is predicted NOT to survive.")

    st.write(
        f"Probability of survival: **{survival_probability:.2f}%**"
    )

    st.write(
        f"Probability of not surviving: **{death_probability:.2f}%**"
    )
