import streamlit as st
import numpy as np
import tensorflow as tf

st.title("Employee Performance Prediction")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("employee_performance_ann.keras")

model = load_model()

training_hours = st.number_input(
    "Training Hours",
    min_value=0.0,
    max_value=100.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

if st.button("Predict Performance"):

    input_data = np.array([
        [training_hours, attendance]
    ], dtype=np.float32)

    probability = model.predict(
        input_data,
        verbose=0
    )[0][0]

    if probability >= 0.5 and training_hours > 5 and attendance > 50:
        prediction = "Good"
    else:
        prediction = "Needs Improvement"

    st.subheader("Prediction")
    st.success(prediction)

    st.write(
        f"Good Probability: {probability * 100:.2f}%"
    )
