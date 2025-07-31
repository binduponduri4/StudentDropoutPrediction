import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🎓 Student Graduation Prediction (Improved Model)")

# ✅ Load static dataset
df = pd.read_csv("dataset.csv")

st.subheader("📄 Raw Data")
st.dataframe(df)

# ✅ Clean and preprocess
df = df[df['Target'] != 'Enrolled']
df['Target'] = df['Target'].map({'Graduate': 1, 'Dropout': 0})

X_full = df.drop('Target', axis=1)
Y = df['Target']

# ✅ Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# ✅ Feature importance using Random Forest
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_scaled, Y)

importances = model_rf.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X_full.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top_10_features = feature_importance['Feature'].head(10).tolist()

# Keep original order in dataset
top_10_features = [col for col in X_full.columns if col in top_10_features]

st.subheader("🔝 Top 10 Features Used")
st.write(top_10_features)

# ✅ Final data for model
X = X_full[top_10_features]
X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# ✅ Evaluation
Y_pred = model.predict(X_test)
report = classification_report(Y_test, Y_pred, output_dict=True)
cm = confusion_matrix(Y_test, Y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {report['accuracy']:.2f}")
st.write(f"**Precision (Graduate):** {report['1']['precision']:.2f}")
st.write(f"**Recall (Graduate):** {report['1']['recall']:.2f}")
st.write(f"**F1 Score (Graduate):** {report['1']['f1-score']:.2f}")
st.write("Confusion Matrix:")
st.write(cm)

# ✅ Plot Feature Importance
st.subheader("📈 Feature Importance (Top 10)")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette="mako")
st.pyplot(plt)

# ✅ Prediction Input
st.subheader("🧮 Predict for New Student (Top 10 Features Only)")
input_data = []

for label in top_10_features:
    unique_vals = df[label].dropna().unique()
    if len(unique_vals) <= 5:
        val = st.selectbox(f"Select {label}:", sorted(unique_vals.tolist()))
    else:
        val = st.number_input(f"Enter {label}:", value=float(df[label].mean()))
    input_data.append(val)

if st.button("🔮 Predict"):
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    result = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    if result == 1:
        st.success(f"✅ Likely to Graduate (Confidence: {proba[1]*100:.2f}%)")
    else:
        st.error(f"❌ Likely to Dropout (Confidence: {proba[0]*100:.2f}%)")
