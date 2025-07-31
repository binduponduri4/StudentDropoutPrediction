import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Student Graduation Predictor", layout="centered")
st.title("🎓 Student Graduation Prediction")

# Load dataset
df = pd.read_csv("dataset.csv")

# Filter for only Dropout and Graduate
df = df[df['Target'] != 'Enrolled']
df['Target'] = df['Target'].map({'Graduate': 1, 'Dropout': 0})

# Split features and target
X_full = df.drop(['Target'], axis=1)
y = df['Target']

# Get feature importances
rf = RandomForestClassifier(random_state=42)
rf.fit(X_full, y)
importances = rf.feature_importances_
importance_series = pd.Series(importances, index=X_full.columns).sort_values(ascending=False)

# Get top 10 important features
top_10_features = list(importance_series.head(10).index)

# Maintain original dataset column order
ordered_top_10 = [col for col in df.columns if col in top_10_features]

# Train model on top 10 features
X = df[ordered_top_10]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# Model accuracy
st.subheader("📊 Model Performance")
acc = accuracy_score(y_test, model.predict(X_test)) * 100
st.write(f"**Accuracy Score:** {acc:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, model.predict(X_test))
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Dropout', 'Graduate'], yticklabels=['Dropout', 'Graduate'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Feature importance chart
st.subheader("🔍 Top 10 Features by Importance")
st.bar_chart(importance_series.head(10))

# User input
st.subheader("🧮 Predict for New Student (Top 10 Inputs Only)")
input_data = []
for feature in ordered_top_10:
    unique_vals = sorted(df[feature].dropna().unique())
    if len(unique_vals) <= 5:
        val = st.selectbox(f"Select {feature}:", unique_vals)
    else:
        val = st.number_input(f"Enter {feature}:", value=float(df[feature].mean()))
    input_data.append(val)

# Predict button
if st.button("🎯 Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    result = "Graduate 🎓" if prediction[0] == 1 else "Dropout ❌"
    st.success(f"The student is likely to: **{result}**")
