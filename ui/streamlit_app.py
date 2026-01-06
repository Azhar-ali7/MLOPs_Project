"""
Streamlit UI for Heart Disease Prediction MLOps Pipeline
Unified interface for testing the complete workflow
"""

import streamlit as st
import requests
import pandas as pd
import json
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuration
# Use environment variable if running in Docker, otherwise localhost
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Heart Disease Prediction - MLOps Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #D4EDDA;
        border: 1px solid #C3E6CB;
        border-radius: 5px;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        background-color: #F8D7DA;
        border: 1px solid #F5C6CB;
        border-radius: 5px;
        color: #721C24;
    }
    .info-box {
        padding: 1rem;
        background-color: #D1ECF1;
        border: 1px solid #BEE5EB;
        border-radius: 5px;
        color: #0C5460;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def make_prediction(patient_data):
    """Make prediction using API"""
    try:
        response = requests.post(f"{API_URL}/predict", json={"data": [patient_data]}, timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def get_metrics():
    """Get Prometheus metrics"""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=2)
        return response.status_code == 200, response.text
    except Exception as e:
        return False, str(e)


def main():
    # Header
    st.markdown('<h1 class="main-header">🫀 Heart Disease Prediction - MLOps Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🔮 Prediction", "📈 Metrics", "🧪 Testing", "📚 Documentation"],
        )

        st.markdown("---")
        st.header("🔍 System Status")

        # Check API health
        api_healthy, health_data = check_api_health()

        if api_healthy:
            st.success("✅ API is Running")
            st.json(health_data)
        else:
            st.error("❌ API is Down")
            st.code("Please start the API:\npython -m uvicorn src.api:app --port 8000")

    # Main content
    if page == "🏠 Home":
        show_home()
    elif page == "🔮 Prediction":
        show_prediction()
    elif page == "📈 Metrics":
        show_metrics()
    elif page == "🧪 Testing":
        show_testing()
    elif page == "📚 Documentation":
        show_documentation()


def show_home():
    """Home page"""
    st.header("Welcome to the Heart Disease Prediction MLOps Platform")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Models Trained", "2", "Random Forest & Logistic Regression")

    with col2:
        st.metric("Accuracy", "90%", "On Test Set")

    with col3:
        st.metric("API Status", "Running" if check_api_health()[0] else "Down")

    st.markdown("---")

    # Quick Access to Monitoring Services
    st.subheader("🔗 Quick Access to Services")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 📚 API Docs")
        st.markdown("[Open Swagger UI](http://localhost:8000/docs)")
        st.caption("Interactive API documentation")
    
    with col2:
        st.markdown("### 📊 MLflow")
        st.markdown("[Open MLflow](http://localhost:5050)")
        st.caption("Experiment tracking & models")
    
    with col3:
        st.markdown("### 📈 Grafana")
        st.markdown("[Open Grafana](http://localhost:3000)")
        st.caption("Metrics dashboards (admin/admin123)")
    
    with col4:
        st.markdown("### 🔍 Prometheus")
        st.markdown("[Open Prometheus](http://localhost:9090)")
        st.caption("Metrics & queries")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 📋 Kibana")
        st.markdown("[Open Kibana](http://localhost:5601)")
        st.caption("Log visualization")
    
    with col2:
        st.markdown("### 🗄️ Elasticsearch")
        st.markdown("[Open Elasticsearch](http://localhost:9200)")
        st.caption("Log storage")
    
    with col3:
        st.markdown("### 🚨 Alertmanager")
        st.markdown("[Open Alertmanager](http://localhost:9093)")
        st.caption("Alert management")
    
    with col4:
        st.markdown("### 🤖 API Health")
        st.markdown("[Check Health](http://localhost:8000/health)")
        st.caption("Backend status")

    st.markdown("---")

    st.subheader("🎯 Getting Started")

    with st.expander("1️⃣ Start the API Server"):
        st.code(
            """
# Option 1: Direct Python
python -m uvicorn src.api:app --port 8000

# Option 2: Using Docker
docker build -t heart-disease-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro heart-disease-api
            """,
            language="bash",
        )

    with st.expander("2️⃣ Make Predictions"):
        st.write("Go to the 🔮 Prediction page to make predictions interactively")

    with st.expander("3️⃣ View Metrics"):
        st.write("Go to the 📈 Metrics page to see API metrics")

    with st.expander("4️⃣ Run Tests"):
        st.write("Go to the 🧪 Testing page to run integration tests")


def show_prediction():
    """Prediction page"""
    st.header("🔮 Heart Disease Prediction")

    # Check API status
    api_healthy, _ = check_api_health()
    if not api_healthy:
        st.error("⚠️ API is not running. Please start it first!")
        return

    st.markdown("Enter patient information below:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=55, help="Patient's age in years")
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic",
        )
        trestbps = st.number_input(
            "Resting Blood Pressure", min_value=80, max_value=200, value=140, help="mm Hg"
        )
        chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=230, help="mg/dl")
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
        )
        restecg = st.selectbox(
            "Resting ECG",
            options=[0, 1, 2],
            help="0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy",
        )

    with col2:
        thalach = st.number_input(
            "Maximum Heart Rate", min_value=60, max_value=220, value=150, help="Beats per minute"
        )
        exang = st.selectbox(
            "Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
        )
        oldpeak = st.number_input(
            "ST Depression", min_value=0.0, max_value=6.0, value=1.0, step=0.1, help="Induced by exercise"
        )
        slope = st.selectbox(
            "Slope of Peak Exercise ST", options=[0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping"
        )
        ca = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3], help="Colored by fluoroscopy")
        thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], help="0: Normal, 1: Fixed Defect, 2: Reversible")

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        predict_button = st.button("🔮 Make Prediction", use_container_width=True, type="primary")

    if predict_button:
        # Prepare data
        patient_data = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }

        with st.spinner("Making prediction..."):
            success, result = make_prediction(patient_data)

        if success:
            st.success("✅ Prediction Complete!")

            # Display results
            prediction = result["predictions"][0]["prediction"]
            probability = result["predictions"][0]["probability"]

            col1, col2 = st.columns(2)

            with col1:
                if prediction == 1:
                    st.markdown(
                        '<div class="error-box"><h3>⚠️ Heart Disease Detected</h3><p>The model predicts presence of heart disease.</p></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="success-box"><h3>✅ No Heart Disease</h3><p>The model predicts no heart disease.</p></div>',
                        unsafe_allow_html=True,
                    )

            with col2:
                st.metric("Confidence", f"{probability:.2%}")
                st.progress(probability)

            # Show input data
            with st.expander("📋 View Input Data"):
                st.json(patient_data)

            # Show raw response
            with st.expander("🔍 View API Response"):
                st.json(result)
        else:
            st.error(f"❌ Prediction Failed: {result.get('error', 'Unknown error')}")


def show_metrics():
    """Metrics page"""
    st.header("📈 API Metrics")

    # Check API status
    api_healthy, _ = check_api_health()
    if not api_healthy:
        st.error("⚠️ API is not running. Please start it first!")
        return

    # Get metrics
    success, metrics = get_metrics()

    if success:
        st.success("✅ Metrics Retrieved")

        # Parse and display key metrics
        st.subheader("🔢 Prometheus Metrics")

        # Display raw metrics
        with st.expander("📊 View Raw Metrics"):
            st.code(metrics, language="text")

        # Parse some key metrics
        st.subheader("📌 Key Metrics")

        lines = metrics.split("\n")
        key_metrics = {}

        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            if "predict_requests_total" in line or "request_duration" in line or "model_inference" in line:
                parts = line.split()
                if len(parts) >= 2:
                    metric_name = parts[0].split("{")[0]
                    metric_value = parts[-1]
                    key_metrics[metric_name] = metric_value

        if key_metrics:
            cols = st.columns(len(key_metrics))
            for idx, (metric, value) in enumerate(key_metrics.items()):
                with cols[idx]:
                    st.metric(metric.replace("_", " ").title(), value)
        else:
            st.info("No metrics available yet. Make some predictions first!")

    else:
        st.error(f"❌ Failed to retrieve metrics: {metrics}")


def show_testing():
    """Testing page"""
    st.header("🧪 Integration Testing")

    # Check API status
    api_healthy, health = check_api_health()

    st.subheader("1️⃣ API Health Check")
    if api_healthy:
        st.success("✅ API is Running")
        st.json(health)
    else:
        st.error("❌ API is Down")
        return

    st.markdown("---")

    st.subheader("2️⃣ Prediction Test")

    if st.button("🧪 Run Prediction Test", type="primary"):
        # Test data
        test_data = {
            "age": 55,
            "sex": 1,
            "cp": 3,
            "trestbps": 140,
            "chol": 230,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 0,
            "ca": 0,
            "thal": 3,
        }

        with st.spinner("Running test..."):
            success, result = make_prediction(test_data)

        if success:
            st.success("✅ Test Passed!")
            st.json(result)
        else:
            st.error(f"❌ Test Failed: {result}")

    st.markdown("---")

    st.subheader("3️⃣ Load Test")

    num_requests = st.slider("Number of Requests", min_value=1, max_value=50, value=10)

    if st.button("🚀 Run Load Test"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        test_data = {
            "age": 55,
            "sex": 1,
            "cp": 3,
            "trestbps": 140,
            "chol": 230,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 0,
            "ca": 0,
            "thal": 3,
        }

        success_count = 0
        total_time = 0

        for i in range(num_requests):
            start_time = time.time()
            success, _ = make_prediction(test_data)
            elapsed = time.time() - start_time

            if success:
                success_count += 1
            total_time += elapsed

            progress_bar.progress((i + 1) / num_requests)
            status_text.text(f"Request {i + 1}/{num_requests} - {elapsed:.3f}s")

        st.success(f"✅ Load Test Complete!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Requests", num_requests)
        with col2:
            st.metric("Successful", success_count)
        with col3:
            st.metric("Avg Response Time", f"{total_time/num_requests:.3f}s")


def show_documentation():
    """Documentation page"""
    st.header("📚 Documentation")

    st.subheader("🎯 Project Overview")
    st.markdown(
        """
    This is a complete MLOps pipeline for heart disease prediction that includes:
    - **Data Processing**: Automated data cleaning and preprocessing
    - **Model Training**: Multiple models with hyperparameter tuning
    - **Experiment Tracking**: MLflow integration
    - **API Serving**: FastAPI with comprehensive endpoints
    - **Monitoring**: Prometheus metrics and structured logging
    - **CI/CD**: GitHub Actions pipeline
    - **Deployment**: Docker and Kubernetes support
    """
    )

    st.subheader("🚀 Quick Commands")

    with st.expander("Start API Server"):
        st.code("python -m uvicorn src.api:app --port 8000 --reload", language="bash")

    with st.expander("Run Tests"):
        st.code("pytest tests/ -v --cov=src", language="bash")

    with st.expander("Train Models"):
        st.code("python -m src.train --data data/processed/heart_processed.csv --model-dir models", language="bash")

    with st.expander("Build Docker Image"):
        st.code("docker build -t heart-disease-api .", language="bash")

    with st.expander("Run Complete Pipeline"):
        st.code("./scripts/run-complete-pipeline.sh", language="bash")

    st.subheader("📊 Model Information")

    model_info = {
        "Random Forest": {"Accuracy": "84.1%", "ROC-AUC": "90.2%", "Best Params": "n_estimators=50, max_depth=5"},
        "Logistic Regression": {
            "Accuracy": "84.1%",
            "ROC-AUC": "91.2%",
            "Best Params": "C=0.1",
        },
    }

    for model, metrics in model_info.items():
        with st.expander(f"📈 {model}"):
            for metric, value in metrics.items():
                st.write(f"**{metric}:** {value}")

    st.subheader("🔗 Useful Links")
    st.markdown(
        """
    - [FastAPI Docs](http://localhost:8000/docs) - Interactive API documentation
    - [GitHub Repository](https://github.com/YOUR_USERNAME/MLOPs_Project)
    - [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
    """
    )


if __name__ == "__main__":
    main()

