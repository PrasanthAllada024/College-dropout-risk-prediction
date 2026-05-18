import streamlit as st
import pandas as pd
import joblib
import warnings

# Remove warnings
warnings.filterwarnings("ignore")

# Page settings
st.set_page_config(
    page_title="Dropout Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align: center;'>🎓 College Dropout Predictor</h1>
<p style='text-align: center; color: gray;'>Predict student dropout risk & analyze behavior</p>
""", unsafe_allow_html=True)

st.divider()

# ---------- LOAD MODEL ----------
model = joblib.load("dropout_risk.pkl")

st.success("Model loaded successfully!")

# ---------- LAYOUT ----------
col1, col2 = st.columns(2)

# ==============================
# ✍️ MANUAL INPUT SECTION
# ==============================
with col1:
    st.subheader("✍️ Single Student Prediction")

    gpa = st.number_input("GPA", 0.0, 10.0, 6.0)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    backlogs = st.number_input("Backlogs", 0, 10, 0)
    study_hours = st.slider("Study Hours per Week", 0, 40, 10)
    engagement = st.slider("Engagement Score", 0, 100, 50)
    income = st.selectbox("Family Income Level", [0, 1, 2])

    if st.button("🔍 Predict Risk"):
        input_data = pd.DataFrame([[
            gpa, attendance, backlogs, study_hours, engagement, income
        ]], columns=[
            "gpa", "attendance", "backlogs", "study_hours", "engagement", "income"
        ])

        prob = model.predict_proba(input_data)[0][1]

        st.subheader("🎯 Prediction Result")
        st.metric("Risk Score", f"{prob:.2f}")

        if prob > 0.7:
            st.error("🔴 High Risk of Dropout")
        elif prob > 0.4:
            st.warning("🟡 Medium Risk of Dropout")
        else:
            st.success("🟢 Low Risk")

# ==============================
# 📂 CSV BULK PREDICTION
# ==============================
with col2:
    st.subheader("📂 Bulk Prediction (Upload CSV)")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("### Uploaded Data")
        st.dataframe(df)

        if st.button("📊 Predict for CSV"):
            probs = model.predict_proba(df)[:, 1]

            df["Risk Score"] = probs

            # Classification logic
            def classify(x):
                if x > 0.7:
                    return "High"
                elif x > 0.4:
                    return "Medium"
                else:
                    return "Low"

            df["Risk Level"] = df["Risk Score"].apply(classify)

            st.write("### 📊 Predictions")
            st.dataframe(df)

            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇ Download Predictions",
                data=csv,
                file_name="dropout_predictions.csv",
                mime="text/csv"
            )
            