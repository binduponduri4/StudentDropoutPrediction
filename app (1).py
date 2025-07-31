import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Graduation Prediction", layout="centered")
st.title("🎓 Student Graduation Prediction (Top 15 Features)")

# ✅ Load static dataset
df = pd.read_csv("dataset.csv")

st.subheader("📄 Raw Data")
st.dataframe(df)

# ✅ Clean & preprocess
df = df[df['Target'] != 'Enrolled']
df['Target'] = df['Target'].map({'Graduate': 1, 'Dropout': 0})

X_full = df.drop('Target', axis=1)
Y = df['Target']

# ✅ Feature selection using RandomForest
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, Y)

importances = rf.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X_full.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top_15_features = feature_importance['Feature'].head(15).tolist()

# Preserve dataset order
top_15_features = [col for col in X_full.columns if col in top_15_features]

st.subheader("🔝 Top 15 Features Used")
st.write(top_15_features)

# ✅ Train model with selected features
X_selected = X_full[top_15_features]
X_scaled_selected = scaler.fit_transform(X_selected)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled_selected, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, Y_train)

# ✅ Model Evaluation
Y_pred = model.predict(X_test)
report = classification_report(Y_test, Y_pred, output_dict=True)
cm = confusion_matrix(Y_test, Y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {report['accuracy']:.2f}")
st.write(f"**Precision (Graduate):** {report['1']['precision']:.2f}")
st.write(f"**Recall (Graduate):** {report['1']['recall']:.2f}")
st.write(f"**F1 Score (Graduate):** {report['1']['f1-score']:.2f}")
st.write("Confusion Matrix:")
st.dataframe(pd.DataFrame(cm, index=["Actual: Dropout", "Actual: Graduate"], columns=["Predicted: Dropout", "Predicted: Graduate"]))

# ✅ Plot Feature Importance
st.subheader("📈 Feature Importance (Top 15)")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15), palette="viridis")
st.pyplot(plt)

# ✅ User input
st.subheader("🧮 Predict for New Student (Top 15 Features Only)")
input_data = []

for feature in top_15_features:
    unique_vals = df[feature].dropna().unique()
    if len(unique_vals) <= 5:
        selected = st.selectbox(f"Select {feature}:", sorted(unique_vals.tolist()))
        input_data.append(selected)
    else:
        default_val = float(df[feature].mean())
        val = st.number_input(f"Enter {feature}:", value=default_val)
        input_data.append(val)

# ✅ Predict
if st.button("🔮 Predict Graduation Status"):
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction] * 100

    if prediction == 1:
        st.success(f"✅ The student is likely to **Graduate** (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"❌ The student is likely to **Dropout** (Confidence: {confidence:.2f}%)")
