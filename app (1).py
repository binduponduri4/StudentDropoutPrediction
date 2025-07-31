import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

st.set_page_config(page_title="Student Graduation Prediction", layout="wide")
st.title("🎓 Student Graduation Prediction (XGBoost Enhanced)")

# ✅ Load static dataset
df = pd.read_csv("dataset.csv")

st.subheader("📄 Raw Data")
st.dataframe(df)

# ✅ Clean + Encode
df = df[df['Target'] != 'Enrolled']
df['Target'] = df['Target'].map({'Graduate': 1, 'Dropout': 0})

X_full = df.drop('Target', axis=1)
Y = df['Target']

# ✅ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# ✅ Use XGBoost for feature importance
xgb_temp = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_temp.fit(X_scaled, Y)

# ✅ Get top 10 features
importance = xgb_temp.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X_full.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

top_10_features = feature_importance['Feature'].head(10).tolist()
top_10_features = [col for col in X_full.columns if col in top_10_features]  # preserve original order

st.subheader("🔝 Top 10 Features Used")
st.write(top_10_features)

# ✅ Prepare final training data
X_top = X_full[top_10_features]
X_top_scaled = scaler.fit_transform(X_top)
X_train, X_test, Y_train, Y_test = train_test_split(X_top_scaled, Y, test_size=0.2, random_state=42)

# ✅ Final model: XGBoost
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.8,
    scale_pos_weight=(Y == 0).sum() / (Y == 1).sum(),  # for imbalance
    random_state=42
)
model.fit(X_train, Y_train)

# ✅ Evaluate
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

# ✅ Feature importance plot
st.subheader("📈 Feature Importance")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette="viridis")
plt.title("Top 10 Feature Importance (XGBoost)")
st.pyplot(plt)

# ✅ Prediction Input
st.subheader("🧮 Predict New Student Outcome")

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
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        st.success(f"✅ Likely to Graduate (Confidence: {confidence[1]*100:.2f}%)")
    else:
        st.error(f"❌ Likely to Dropout (Confidence: {confidence[0]*100:.2f}%)")
