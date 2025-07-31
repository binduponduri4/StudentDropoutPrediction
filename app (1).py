import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Student Graduation Predictor", layout="centered")

st.title("🎓 Student Graduation Prediction")

# Load dataset from local file
df = pd.read_csv("dataset.csv")

# Filter and prepare the target column
df = df[df['Target'] != 'Enrolled']
df['Target'] = df['Target'].map({'Graduate': 1, 'Dropout': 0})

# Split features and labels
X = df.drop(['Target'], axis=1)
Y = df['Target']

# Feature importance with Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X, Y)
importances = rf.feature_importances_
feature_scores = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Top 10 features
top_10_features = feature_scores.head(10).index.tolist()
X_top = X[top_10_features]

# Train model
X_train, X_test, Y_train, Y_test = train_test_split(X_top, Y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=4000)
model.fit(X_train, Y_train)

# Accuracy and Confusion Matrix
pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, pred) * 100
cm = confusion_matrix(Y_test, pred)

# Show performance
st.subheader("📊 Model Performance")
st.write(f"**Accuracy Score:** {accuracy:.2f}%")

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Dropout', 'Graduate'], yticklabels=['Dropout', 'Graduate'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Feature Importance Plot
st.subheader("🔍 Top 10 Important Features")
st.bar_chart(feature_scores.head(10))

# Input section
st.subheader("🧮 Predict for New Student (Top 10 Inputs Only)")

input_data = []
for feature in top_10_features:
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
