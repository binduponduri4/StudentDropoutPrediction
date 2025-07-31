import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🎓 Student Graduation Prediction (Top 10 Features)")

# ✅ Step 1: Load static dataset (no upload needed)
df = pd.read_csv("dataset.csv")  # <-- make sure this file is in the same directory

st.subheader("📄 Raw Data")
st.dataframe(df)

# ✅ Step 2: Preprocessing
df = df[df['Target'] != 'Enrolled']
df['Target'] = df['Target'].map({'Graduate': 1, 'Dropout': 0})

X_full = df.drop(['Target'], axis=1)
Y = df['Target']

# ✅ Step 3: Train temporary model for feature selection
temp_model = LogisticRegression(max_iter=4000)
temp_model.fit(X_full, Y)

importance = np.abs(temp_model.coef_[0])
feature_importance = pd.DataFrame({
    'Feature': X_full.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

top_10_features = feature_importance['Feature'].head(10).tolist()

st.subheader("🔝 Top 10 Most Important Features")
st.write(top_10_features)

# ✅ Step 4: Use top 10 features for training and input
X = X_full[top_10_features]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=4000)
model.fit(X_train, Y_train)

# ✅ Step 5: Evaluation
pred = model.predict(X_test)
accuracy = accuracy_score(pred, Y_test) * 100
cm = confusion_matrix(pred, Y_test)

st.subheader("📊 Model Performance")
st.write(f"**Accuracy Score:** {accuracy:.2f}%")
st.write("**Confusion Matrix:**")
st.write(cm)

# ✅ Step 6: Feature Importance Chart
st.subheader("📈 Feature Importance (Top 10)")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette="mako")
plt.title("Top 10 Feature Importance")
st.pyplot(plt)

# ✅ Step 7: New prediction input
st.subheader("🧮 Predict for New Student (Top 10 Inputs Only)")

input_data = []

for label in top_10_features:
    unique_vals = df[label].dropna().unique()

    # Use dropdown if binary/categorical
    if len(unique_vals) <= 5:
        options = sorted(unique_vals.tolist())
        selected_val = st.selectbox(f"Select {label}:", options)
        input_data.append(selected_val)
    else:
        val = st.number_input(f"Enter {label}:", value=float(df[label].mean()))
        input_data.append(val)

if st.button("🔮 Predict"):
    new_data = np.array([input_data])
    if len(new_data[0]) != len(top_10_features):
        st.error(f"Expected {len(top_10_features)} features, but got {len(new_data[0])}")
    else:
        label = model.predict(new_data)
        if label[0] == 0:
            st.success("❌ The student is likely to **Dropout**.")
        else:
            st.success("✅ The student is likely to **Graduate**.")
