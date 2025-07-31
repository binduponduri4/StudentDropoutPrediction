import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🎓 Student Graduation Prediction using XGBoost (Top 10 Features)")

# ✅ Load static dataset
df = pd.read_csv("dataset.csv")

st.subheader("📄 Raw Data")
st.dataframe(df)

# ✅ Preprocessing
df = df[df['Target'] != 'Enrolled']
df['Target'] = df['Target'].map({'Graduate': 1, 'Dropout': 0})

X_full = df.drop('Target', axis=1)
Y = df['Target']

# ✅ Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# ✅ Feature selection using XGBoost
xgb_temp = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_temp.fit(X_scaled, Y)

importances = xgb_temp.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X_full.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top_10_features = feature_importance['Feature'].head(10).tolist()
# Keep original order in dataset
top_10_features = [col for col in X_full.columns if col in top_10_features]

st.subheader("🔝 Top 10 Features Used")
st.write(top_10_features)

# ✅ Train final model on top 10 features
X_top = X_full[top_10_features]
X_scaled_top = scaler.fit_transform(X_top)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled_top, Y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, Y_train)

# ✅ Evaluate model
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

# ✅ Feature Importance Bar Plot
st.subheader("📈 Feature Importance (Top 10)")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette="viridis")
st.pyplot(plt)

# ✅ Input for prediction
st.subheader("🧮 Predict for New Student (Top 10 Features Only)")
input_data = []

for label in top_10_features:
    unique_vals = df[label].dropna().unique()
    if len(unique_vals) <= 5:
        val = st.selectbox(f"Select {label}:", sorted(unique_vals.tolist()))
    else:
        val = st.number_input(f"Enter {label}:", value=float(df[label].mean()))
    input_data.append(val)

if st.button("🔮 Predict Graduation Status"):
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    result = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    if result == 1:
        st.success(f"✅ The student is likely to **Graduate** (Confidence: {proba[1]*100:.2f}%)")
    else:
        st.error(f"❌ The student is likely to **Dropout** (Confidence: {proba[0]*100:.2f}%)")
