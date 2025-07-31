import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🎓 Student Graduation Prediction App")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📄 Raw Data")
    st.dataframe(df)

    # Preprocess
    df = df[df['Target'] != 'Enrolled']
    df['Target'] = df['Target'].map({'Graduate': 1, 'Dropout': 0})

    X = df.drop(['Target'], axis=1)
    Y = df['Target']

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=4000)
    model.fit(X_train, Y_train)

    # Evaluate
    pred = model.predict(X_test)
    accuracy = accuracy_score(pred, Y_test) * 100
    cm = confusion_matrix(pred, Y_test)

    st.subheader("📊 Model Performance")
    st.write(f"**Accuracy Score:** {accuracy:.2f}%")
    st.write("**Confusion Matrix:**")
    st.write(cm)

    # Feature Importance
    importance = np.abs(model.coef_[0])
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    st.subheader("🔍 Feature Importance (Table)")
    st.dataframe(feature_importance)

    st.subheader("📈 Feature Importance (Bar Chart)")
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15), palette="viridis")
    plt.title("Top 15 Most Important Features")
    st.pyplot(plt)

    # Prediction section
    st.subheader("🧮 Predict for New Student")

    input_data = []
    input_labels = list(X.columns)

    for label in input_labels:
        val = st.number_input(f"Enter {label}:", value=0.0)
        input_data.append(val)

    if st.button("🔮 Predict"):
        new_data = np.array([input_data])

        if len(new_data[0]) != X.shape[1]:
            st.error(f"Expected {X.shape[1]} inputs, but got {len(new_data[0])}.")
        else:
            label = model.predict(new_data)
            if label[0] == 0:
                st.success("❌ The student is likely to **Dropout**.")
            else:
                st.success("✅ The student is likely to **Graduate**.")
