import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("Student Graduation Prediction")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Raw Data")
    st.dataframe(df)

    df = df[df['Target'] != 'Enrolled']
    df['Target'] = df['Target'].map({'Graduate': 1, 'Dropout': 0})

    X = df.drop(['Target'], axis=1)
    Y = df['Target']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=4000)
    model.fit(X_train, Y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(pred, Y_test) * 100
    cm = confusion_matrix(pred, Y_test)

    st.subheader("Model Performance")
    st.write(f"Accuracy Score: **{accuracy:.2f}%**")
    st.write("Confusion Matrix:")
    st.write(cm)

    st.subheader("Predict for New Data")

    input_data = []
    input_labels = [
        "Marital Status", "Age at enrollment", "International Student (1-yes/0-no)", "Curricular Units 1st sem (credited)",
        "Gender (1-Male/0-Female)", "Scholarship Holder (1-yes/0-no)", "Curricular Units 2nd sem (credited)",
        "Displaced", "Educational Special Needs", "Debtor", "Tuition fees up to date", "Gender (again if needed)", 
        "Unemployment Rate", "Previous Qualification (0-none/1-yes)", "Previous Qualification Grade", 
        "Application Mode", "Application Order", "Mother Qualification", "Father Qualification",
        "Admission Grade", "Displaced Again", "Previous Qualification (yes/no again)", "Mother Occupation",
        "Father Occupation", "Curricular Units 1st sem (enrolled)", "Curricular Units 1st sem (approved)",
        "Curricular Units 2nd sem (enrolled)", "Curricular Units 2nd sem (approved)",
        "Curricular Units 1st sem (grade)", "Curricular Units 2nd sem (grade)", 
        "Curricular Units 2nd sem (evaluations)", "Curricular Units 1st sem (evaluations)", 
        "Average Grade", "Unemployment rate change", "GDP"

    ]

    for label in input_labels:
        val = st.number_input(f"Enter {label}:", value=0.0)
        input_data.append(val)

    if st.button("Predict"):
        new_data = np.array([input_data])

        if len(new_data[0]) != X.shape[1]:
            st.error(f"Expected {X.shape[1]} features but got {len(new_data[0])}. Please enter correct number of inputs.")
        else:
            label = model.predict(new_data)
            if label[0] == 0:
                st.success("The student is likely to Dropout.")
            else:
                st.success("The student is likely to Graduate.")
