import streamlit as st
import numpy as np
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="💰",
    layout="centered"
)

# ---------------- Custom Styling ----------------
st.markdown("""
    <style>
    body {
        background-color: #0E1117;
    }
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        color: black;
        font-size: 18px;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .prediction-card {
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1f1c2c, #928dab);
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        color: white;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.4);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
model = joblib.load("insurance_model.pkl")

# ---------------- Title ----------------
st.title("💰 Insurance Charges Predictor")
st.write("### Estimate your medical insurance cost instantly")

st.divider()

# ---------------- Layout ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 30)
    bmi = st.slider("BMI", 15.0, 45.0, 25.0)
    children = st.slider("Number of Children", 0, 5, 0)

with col2:
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    sex = st.selectbox("Sex", ["Female", "Male"])
    region_southeast = st.selectbox("Region Southeast", ["No", "Yes"])
    bmi_obese = st.selectbox("Obese BMI Category", ["No", "Yes"])

# ---------------- Encoding ----------------
encoded_smoker = 1 if smoker == "Yes" else 0
encoded_sex = 1 if sex == "Male" else 0
region_southeast = 1 if region_southeast == "Yes" else 0
bmi_category_Obese = 1 if bmi_obese == "Yes" else 0

# ---------------- Input Array ----------------
input_data = np.array([[age, bmi, children,
                        encoded_smoker,
                        region_southeast,
                        encoded_sex,
                        bmi_category_Obese]])

st.divider()

# ---------------- Prediction ----------------
if st.button("🚀 Predict Insurance Charges"):
    with st.spinner("Calculating prediction..."):

        prediction_log = model.predict(input_data)
        prediction = np.exp(prediction_log)

    # Prediction Card
    st.markdown(f"""
        <div class="prediction-card">
            Estimated Insurance Cost <br><br>
            💵 ${prediction[0]:,.2f}
        </div>
    """, unsafe_allow_html=True)

    # Risk Indicator
    if prediction[0] < 10000:
        st.success("🟢 Low insurance cost range")
    elif prediction[0] < 30000:
        st.warning("🟡 Moderate insurance cost range")
    else:
        st.error("🔴 High insurance cost range")

    if encoded_smoker == 1:
        st.info("⚠ Smoking significantly increases insurance charges.")