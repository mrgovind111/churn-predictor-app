import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Customer Churn Predictor")
st.caption("Enter customer details to predict churn risk.")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("API URL", value=API_URL)
    st.caption("Change this after deploying to the cloud.")

with st.form("churn_form"):
    st.subheader("Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0, max_value=100, value=12, step=1,
        )
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0, max_value=500.0, value=70.0, step=1.0,
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0, max_value=10000.0, value=850.0, step=10.0,
        )

    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

if submitted:
    payload = {
        "tenure": int(tenure),
        "monthly_charges": float(monthly_charges),
        "total_charges": float(total_charges),
        "gender": gender,
    }

    try:
        with st.spinner("Calling the model..."):
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()

        st.divider()
        if result["prediction"] == 1:
            st.error(f"⚠️ **{result['label']}**")
        else:
            st.success(f"✅ **{result['label']}**")

        colA, colB = st.columns(2)
        colA.metric("Probability", f"{result['probability']:.1%}")
        colB.metric("Class", result["label"])

        with st.expander("Raw JSON response"):
            st.json(result)

    except requests.exceptions.ConnectionError:
        st.error(f"❌ Could not reach the API at `{api_url}`. Is the backend running?")
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The API took too long to respond.")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
