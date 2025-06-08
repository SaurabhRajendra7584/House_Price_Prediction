import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ✅ Page config
st.set_page_config(page_title="House Price Predictor", layout="wide")

# ✅ Custom Header with Your Name
st.markdown("""
# 🏠 House Price Prediction App

### 👨‍💻 Developed by **Saurabh Rajendra**  
🔗 GitHub: https://github.com/SaurabhRajendra7584

_Predict housing prices using a powerful XGBoost model._

---
""")

# ✅ Load Model
model_path = 'model/house_price_model_xgboost.pkl'
if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please ensure it's in the `model/` directory.")
    st.stop()

model = joblib.load(model_path)

# ✅ Load Dataset
data_path = 'data/cleaned_encoded_house_data.csv'
if not os.path.exists(data_path):
    st.error("❌ Data file not found. Please ensure it's in the `data/` directory.")
    st.stop()

df = pd.read_csv(data_path)

# ✅ Input Section
st.subheader("📋 Enter Property Details:")
sample_input = df.drop('SalePrice', axis=1).iloc[0:1]
user_input = {}

for col in sample_input.columns:
    if df[col].nunique() <= 10:
        user_input[col] = st.selectbox(f"{col}", options=sorted(df[col].unique()))
    else:
        user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))

input_df = pd.DataFrame([user_input])

# ✅ Prediction
if st.button("🔍 Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Price: ₹{prediction:,.2f}")

# ✅ Sidebar
st.sidebar.header("📊 Data Overview")
if st.sidebar.checkbox("Show Raw Data"):
    st.sidebar.write(df.head(100))

st.sidebar.header("🔎 Feature Insights")
top_feats = df.corr()['SalePrice'].abs().sort_values(ascending=False)[1:6]
fig, ax = plt.subplots()
sns.barplot(x=top_feats.values, y=top_feats.index, ax=ax)
ax.set_title("Top 5 Features Correlated with Sale Price")
st.sidebar.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.markdown("🔗 GitHub: https://github.com/SaurabhRajendra7584")
st.sidebar.markdown("👨‍💻 Powered by Streamlit + XGBoost")

# ✅ Footer (keep as-is)
st.markdown("---")
st.markdown("© 2025 — Developed by [Saurabh Rajendra](https://github.com/SaurabhRajendra7584)")
