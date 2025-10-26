import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ------------------- LOAD MODELS & PREPROCESSORS -------------------
with open("preprocessor1.pkl", "rb") as f:
    preprocessor1 = pickle.load(f)

with open("pts_model.pkl", "rb") as f:
    pts_model = pickle.load(f)

with open("preprocessor2.pkl", "rb") as f:
    preprocessor2 = pickle.load(f)

with open("roi_model.pkl", "rb") as f:
    roi_model = pickle.load(f)

st.title("📊 Training Effectiveness & ROI Prediction App")

st.write("Enter Employee and Training Details to Estimate:")
st.write("- Post-Training Score")
st.write("- Performance Gain")
st.write("- Return on Investment (ROI)")

# ------------------- INPUT FORM -------------------
with st.form("input_form"):
    Department = st.selectbox("Department", ['Finance', 'Marketing', 'HR', 'IT', 'Sales'])
    Role = st.selectbox("Role", ['Executive', 'Analyst', 'Manager', 'Team Lead'])
    ProgramID = st.selectbox("Program ID", ['P100', 'P101', 'P102', 'P103', 'P104', 'P105', 'P106'])
    ProgramDifficulty = st.selectbox("Program Difficulty", ["Low", "Medium", "High"])
    DeliveryMode = st.selectbox("Delivery Mode", ["Online", "In-Person", "Hybrid"])

    Tenure = st.number_input("Tenure (Years)", min_value=0.0, step=0.1)
    Salary = st.number_input("Salary", min_value=0.0, step=0.1)
    PreviousTrainings = st.number_input("Previous Trainings Completed", min_value=0, step=1)
    DepartmentBudget = st.number_input("Department Budget")
    ProgramCost = st.number_input("Program Cost")
    TrainingHours = st.number_input("Training Hours", min_value=1.0, step=0.5)
    InstructorRating = st.number_input("Instructor Rating (1-5)", min_value=1.0, max_value=5.0, step=0.1)
    MonthsSinceLastTraining = st.number_input("Months Since Last Training", min_value=0, step=1)
    PreTrainingScore = st.number_input("Pre-Training Score (0-100)", min_value=0.0, max_value=100.0, step=0.5)

    TrainingDate = st.date_input("Training Date")

    submitted = st.form_submit_button("Predict")

# ------------------- PROCESS & PREDICT -------------------
if submitted:
    dataset = pd.DataFrame([{
        "Department": Department,
        "Role": Role,
        "ProgramID": ProgramID,
        "ProgramDifficulty": ProgramDifficulty,
        "DeliveryMode": DeliveryMode,
        "Tenure": Tenure,
        "Salary": Salary,
        "PreviousTrainings": PreviousTrainings,
        "DepartmentBudget": DepartmentBudget,
        "ProgramCost": ProgramCost,
        "TrainingHours": TrainingHours,
        "InstructorRating": InstructorRating,
        "MonthsSinceLastTraining": MonthsSinceLastTraining,
        "PreTrainingScore": PreTrainingScore,
        "TrainingDate": pd.to_datetime(TrainingDate)
    }])

    # Feature Engineering
    dataset["month"] = dataset["TrainingDate"].dt.month
    dataset['month_sin'] = np.sin(2 * np.pi * dataset['month'] / 12)
    dataset['month_cos'] = np.cos(2 * np.pi * dataset['month'] / 12)

    dataset["budget_ratio"] = dataset["ProgramCost"] / dataset["DepartmentBudget"]

    # Model 1: Predict Post Training Score
    X1 = preprocessor1.transform(dataset)
    post_training_score_pred = pts_model.predict(X1)[0]

    # Performance Gain
    performance_gain = post_training_score_pred - PreTrainingScore
    dataset["preformance_gain"] = performance_gain

    # Model 2: Predict ROI
    X2 = preprocessor2.transform(dataset)
    roi_pred = roi_model.predict(X2)[0]

    # ------------------- DISPLAY RESULTS -------------------
    st.subheader("✅ Predictions")
    st.write(f"**Predicted Post-Training Score:** {post_training_score_pred:.2f}")
    st.write(f"**Performance Gain:** {performance_gain:.2f}")
    st.write(f"**Predicted ROI:** {roi_pred:.2f}")
