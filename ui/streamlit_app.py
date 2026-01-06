"""
Streamlit UI for Heart Disease Prediction MLOps Pipeline
Unified interface for testing the complete workflow with MLflow integration
"""

import os
import sys
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_and_preprocess_data  # noqa: E402

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")

# Don't set MLflow tracking URI at module level - do it lazily
MLFLOW_CONFIGURED = False


def configure_mlflow():
    """Configure MLflow lazily to avoid blocking on import"""
    global MLFLOW_CONFIGURED
    if not MLFLOW_CONFIGURED:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            MLFLOW_CONFIGURED = True
        except Exception:
            pass
    return MLFLOW_CONFIGURED


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
    st.markdown(
        '<h1 class="main-header">' "🫀 Heart Disease Prediction - MLOps Dashboard" "</h1>", unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        page = st.radio(
            "Select Page",
            [
                "🏠 Home",
                "🤖 Train Models",
                "🔬 MLflow Experiments",
                "🔮 Prediction",
                "📈 Metrics",
                "🧪 Testing",
                "📚 Documentation",
            ],
        )

        st.markdown("---")
        st.header("🔍 Quick Links")
        st.markdown("- [API Docs](http://localhost:8000/docs)")
        st.markdown("- [MLflow](http://localhost:5050)")
        st.markdown("- [Grafana](http://localhost:3000)")
        st.markdown("- [Prometheus](http://localhost:9090)")

    # Main content
    if page == "🏠 Home":
        show_home()
    elif page == "🤖 Train Models":
        show_training()
    elif page == "🔬 MLflow Experiments":
        show_mlflow()
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
        api_status = "Running" if check_api_health()[0] else "Down"
        st.metric("API Status", api_status)

    st.markdown("---")

    # Quick Access to Services
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
        st.caption("Metrics dashboards (admin/admin)")

    with col4:
        st.markdown("### 🔍 Prometheus")
        st.markdown("[Open Prometheus](http://localhost:9090)")
        st.caption("Metrics & queries")

    st.markdown("---")

    st.subheader("🎯 Getting Started")

    with st.expander("1️⃣ Train Models"):
        st.write("Go to the 🤖 Train Models page to train models " "with MLflow tracking and hyperparameter tuning")

    with st.expander("2️⃣ View Experiments"):
        st.write("Go to the 🔬 MLflow Experiments page to " "compare model runs")

    with st.expander("3️⃣ Make Predictions"):
        st.write("Go to the 🔮 Prediction page to make predictions " "interactively")


def show_training():
    """Model training page with MLflow integration"""
    st.header("🤖 Model Training with MLflow")

    st.markdown(
        """
    Train machine learning models with automatic experiment tracking
    using MLflow. All parameters, metrics, and artifacts are logged
    for reproducibility.
    """
    )

    # Data loading
    st.subheader("📁 Data Configuration")

    col1, col2 = st.columns(2)
    with col1:
        data_path = st.text_input(
            "Data Path", value="data/processed/heart_processed.csv", help="Path to processed CSV file"
        )

    with col2:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05, help="Proportion of data for testing")

    # Model selection
    st.subheader("🎯 Model Configuration")

    model_type = st.selectbox(
        "Select Model", ["Random Forest", "Logistic Regression", "Both"], help="Choose which model(s) to train"
    )

    # Hyperparameter tuning option
    tune_hyperparams = st.checkbox(
        "Enable Hyperparameter Tuning", value=False, help="Use GridSearchCV to find best parameters"
    )

    cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5, help="Number of CV folds")

    st.markdown("---")

    # Model-specific hyperparameters
    if model_type in ["Random Forest", "Both"]:
        st.subheader("🌲 Random Forest Parameters")

        if tune_hyperparams:
            st.markdown("**Parameter Grid for Tuning:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                rf_n_estimators = st.multiselect("n_estimators", [10, 50, 100, 200], default=[50, 100])
            with col2:
                rf_max_depth = st.multiselect(
                    "max_depth",
                    [None, 5, 10, 20],
                    default=[5, 10],
                    format_func=lambda x: ("None" if x is None else str(x)),
                )
            with col3:
                rf_min_samples_split = st.multiselect("min_samples_split", [2, 5, 10], default=[2, 5])
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                rf_n_estimators = st.number_input("n_estimators", 10, 500, 100)
            with col2:
                rf_max_depth = st.number_input("max_depth", 1, 50, 10)
            with col3:
                rf_min_samples_split = st.number_input("min_samples_split", 2, 20, 2)

    if model_type in ["Logistic Regression", "Both"]:
        st.subheader("📊 Logistic Regression Parameters")

        if tune_hyperparams:
            st.markdown("**Parameter Grid for Tuning:**")
            col1, col2 = st.columns(2)
            with col1:
                lr_C = st.multiselect("C (Regularization)", [0.01, 0.1, 1.0, 10.0], default=[0.1, 1.0])
            with col2:
                lr_penalty = st.multiselect("Penalty", ["l2"], default=["l2"])
        else:
            col1, col2 = st.columns(2)
            with col1:
                lr_C = st.number_input("C (Regularization)", 0.01, 100.0, 1.0, step=0.1)
            with col2:
                lr_penalty = st.selectbox("Penalty", ["l2"])

    # MLflow experiment name
    st.subheader("📊 MLflow Configuration")
    experiment_name = st.text_input("Experiment Name", value="heart-disease-prediction", help="MLflow experiment name")

    st.markdown("---")

    # Training button
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        if not Path(data_path).exists():
            st.error(f"❌ Data file not found: {data_path}")
            return

        # Configure and set MLflow experiment
        configure_mlflow()
        mlflow.set_experiment(experiment_name)

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Load data
            status_text.text("📥 Loading data...")
            progress_bar.progress(0.1)

            data_result = load_and_preprocess_data(data_path, test_size=test_size)
            X_train, X_test, y_train, y_test, feature_names = data_result

            st.success(f"✅ Data loaded: {len(X_train)} training samples, " f"{len(X_test)} test samples")

            models_to_train = []
            if model_type == "Both":
                models_to_train = ["Random Forest", "Logistic Regression"]
            else:
                models_to_train = [model_type]

            results = {}

            for idx, model_name in enumerate(models_to_train):
                status_text.text(f"🔧 Training {model_name}...")
                progress_bar.progress(0.2 + (idx * 0.4))

                run_name = f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
                with mlflow.start_run(run_name=run_name):
                    # Log parameters
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("test_size", test_size)
                    mlflow.log_param("cv_folds", cv_folds)
                    mlflow.log_param("tune_hyperparams", tune_hyperparams)
                    mlflow.log_param("data_path", data_path)

                    # Train model
                    if model_name == "Random Forest":
                        if tune_hyperparams:
                            param_grid = {
                                "n_estimators": rf_n_estimators,
                                "max_depth": rf_max_depth,
                                "min_samples_split": rf_min_samples_split,
                            }
                            model = GridSearchCV(
                                RandomForestClassifier(random_state=42),
                                param_grid,
                                cv=cv_folds,
                                scoring="roc_auc",
                                n_jobs=-1,
                            )
                            model.fit(X_train, y_train)
                            best_model = model.best_estimator_
                            mlflow.log_params(model.best_params_)
                            mlflow.log_metric("best_cv_score", model.best_score_)
                        else:
                            n_est = rf_n_estimators if isinstance(rf_n_estimators, int) else rf_n_estimators[0]
                            max_d = rf_max_depth if isinstance(rf_max_depth, int) else rf_max_depth[0]
                            min_split = (
                                rf_min_samples_split
                                if isinstance(rf_min_samples_split, int)
                                else rf_min_samples_split[0]
                            )
                            best_model = RandomForestClassifier(
                                n_estimators=n_est,
                                max_depth=max_d,
                                min_samples_split=min_split,
                                random_state=42,
                            )
                            mlflow.log_param("n_estimators", best_model.n_estimators)
                            mlflow.log_param("max_depth", best_model.max_depth)
                            mlflow.log_param("min_samples_split", best_model.min_samples_split)
                            best_model.fit(X_train, y_train)

                    else:  # Logistic Regression
                        if tune_hyperparams:
                            param_grid = {"C": lr_C, "penalty": lr_penalty}
                            model = GridSearchCV(
                                LogisticRegression(random_state=42, max_iter=1000),
                                param_grid,
                                cv=cv_folds,
                                scoring="roc_auc",
                                n_jobs=-1,
                            )
                            model.fit(X_train, y_train)
                            best_model = model.best_estimator_
                            mlflow.log_params(model.best_params_)
                            mlflow.log_metric("best_cv_score", model.best_score_)
                        else:
                            c_val = lr_C if isinstance(lr_C, (int, float)) else lr_C[0]
                            pen_val = lr_penalty if isinstance(lr_penalty, str) else lr_penalty[0]
                            best_model = LogisticRegression(
                                C=c_val,
                                penalty=pen_val,
                                random_state=42,
                                max_iter=1000,
                            )
                            mlflow.log_param("C", best_model.C)
                            mlflow.log_param("penalty", best_model.penalty)
                            best_model.fit(X_train, y_train)

                    # Predictions
                    y_pred = best_model.predict(X_test)
                    y_proba = best_model.predict_proba(X_test)[:, 1]

                    # Metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    roc_auc = roc_auc_score(y_test, y_proba)

                    # Cross-validation score
                    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring="roc_auc")
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()

                    # Log metrics
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("roc_auc", roc_auc)
                    mlflow.log_metric("cv_mean", cv_mean)
                    mlflow.log_metric("cv_std", cv_std)

                    # Save model (with fallback for API compatibility)
                    try:
                        mlflow.sklearn.log_model(best_model, "model", registered_model_name=None)
                    except Exception as e:
                        st.warning(f"Model logging skipped due to MLflow API: {str(e)}")
                        # Save model locally as fallback
                        import joblib

                        model_path = f"/app/models/{model_name.lower().replace(' ', '_')}.joblib"
                        joblib.dump(best_model, model_path)
                        mlflow.log_artifact(model_path)

                    # Save feature names
                    mlflow.log_dict({"features": feature_names}, "features.json")

                    results[model_name] = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "roc_auc": roc_auc,
                        "cv_mean": cv_mean,
                        "cv_std": cv_std,
                        "run_id": mlflow.active_run().info.run_id,
                    }

            progress_bar.progress(1.0)
            status_text.text("✅ Training complete!")

            st.success("🎉 Training completed successfully!")

            # Display results
            st.subheader("📊 Training Results")

            for model_name, metrics in results.items():
                with st.expander(f"📈 {model_name} Results", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.4f}")
                    with col4:
                        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

                    st.write(f"**CV Score:** {metrics['cv_mean']:.4f} " f"(±{metrics['cv_std']:.4f})")
                    st.write(f"**MLflow Run ID:** `{metrics['run_id']}`")

            st.info("💡 View detailed results in the MLflow UI: " "http://localhost:5050")

        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            import traceback

            st.code(traceback.format_exc())


def show_mlflow():
    """MLflow experiments page"""
    st.header("🔬 MLflow Experiments")

    st.markdown(
        """
    View and compare machine learning experiments tracked with MLflow.
    """
    )

    # Configure MLflow lazily
    configure_mlflow()

    try:
        # Get all experiments with timeout
        import socket

        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(5)
        try:
            experiments = mlflow.search_experiments()
        finally:
            socket.setdefaulttimeout(old_timeout)

        if not experiments:
            st.warning("No experiments found. Train a model first!")
            return

        # Experiment selector
        exp_names = [exp.name for exp in experiments]
        selected_exp = st.selectbox("Select Experiment", exp_names)

        # Get experiment
        experiment = mlflow.get_experiment_by_name(selected_exp)

        st.subheader(f"📊 Experiment: {experiment.name}")
        st.write(f"**Experiment ID:** {experiment.experiment_id}")
        st.write(f"**Artifact Location:** {experiment.artifact_location}")

        # Get runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if runs.empty:
            st.info("No runs found in this experiment.")
            return

        st.write(f"**Total Runs:** {len(runs)}")

        # Display runs table
        st.subheader("🏃 Recent Runs")

        # Select columns to display
        display_cols = []
        if "start_time" in runs.columns:
            display_cols.append("start_time")
        if "tags.mlflow.runName" in runs.columns:
            display_cols.append("tags.mlflow.runName")

        metric_cols = [col for col in runs.columns if col.startswith("metrics.")]
        param_cols = [col for col in runs.columns if col.startswith("params.")]

        display_cols.extend(metric_cols[:10])
        display_cols.extend(param_cols[:5])

        if display_cols:
            st.dataframe(runs[display_cols].head(10), use_container_width=True)
        else:
            st.dataframe(runs.head(10), use_container_width=True)

        # Compare top runs
        st.subheader("🏆 Top Performing Runs")

        metric_options = [col.replace("metrics.", "") for col in metric_cols] if metric_cols else []
        metric_to_sort = st.selectbox("Sort by Metric", metric_options)

        if metric_to_sort:
            sorted_runs = runs.sort_values(f"metrics.{metric_to_sort}", ascending=False)

            top_n = st.slider("Number of runs to show", 1, min(10, len(sorted_runs)), 5)

            # Display top runs
            for idx, (_, run) in enumerate(sorted_runs.head(top_n).iterrows(), 1):
                run_name = run.get("tags.mlflow.runName", run["run_id"][:8])
                with st.expander(f"#{idx} Run: {run_name}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Metrics:**")
                        for col in metric_cols:
                            metric_name = col.replace("metrics.", "")
                            if not pd.isna(run[col]):
                                st.write(f"- {metric_name}: " f"{run[col]:.4f}")

                    with col2:
                        st.write("**Parameters:**")
                        for col in param_cols:
                            param_name = col.replace("params.", "")
                            if not pd.isna(run[col]):
                                st.write(f"- {param_name}: {run[col]}")

                    st.write(f"**Run ID:** `{run['run_id']}`")

        st.markdown("---")
        st.info("💡 For detailed analysis, open the MLflow UI: " "http://localhost:5050")

    except Exception as e:
        st.error(f"❌ Failed to load experiments: {str(e)}")
        st.info("Make sure MLflow is running at: " + MLFLOW_TRACKING_URI)


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
        age = st.number_input("Age", min_value=1, max_value=120, value=55, help="Patient's age in years")
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: ("Female" if x == 0 else "Male"))
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            help=("0: Typical Angina, 1: Atypical Angina, " "2: Non-anginal Pain, 3: Asymptomatic"),
        )
        trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=140, help="mm Hg")
        chol = st.number_input("Cholesterol", min_value=50, max_value=600, value=230, help="mg/dl")
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
        )
        restecg = st.selectbox(
            "Resting ECG",
            options=[0, 1, 2],
            help=("0: Normal, 1: ST-T Wave Abnormality, " "2: Left Ventricular Hypertrophy"),
        )

    with col2:
        thalach = st.number_input("Maximum Heart Rate", min_value=30, max_value=250, value=150, help="Beats per minute")
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input(
            "ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="Induced by exercise"
        )
        slope = st.selectbox(
            "Slope of Peak Exercise ST", options=[0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping"
        )
        ca = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3, 4], help="Colored by fluoroscopy")
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
                        '<div class="error-box">'
                        "<h3>⚠️ Heart Disease Detected</h3>"
                        "<p>The model predicts presence of "
                        "heart disease.</p></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="success-box">'
                        "<h3>✅ No Heart Disease</h3>"
                        "<p>The model predicts no heart disease."
                        "</p></div>",
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
            st.error(f"❌ Prediction Failed: " f"{result.get('error', 'Unknown error')}")


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
            st.info("No metrics available yet. " "Make some predictions first!")

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
            status_text.text(f"Request {i + 1}/{num_requests} - " f"{elapsed:.3f}s")

        st.success("✅ Load Test Complete!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Requests", num_requests)
        with col2:
            st.metric("Successful", success_count)
        with col3:
            st.metric("Avg Response Time", f"{total_time / num_requests:.3f}s")


def show_documentation():
    """Documentation page"""
    st.header("📚 Documentation")

    st.subheader("🎯 Project Overview")
    st.markdown(
        """
    This is a complete MLOps pipeline for heart disease prediction
    that includes:
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

    with st.expander("Run Tests"):
        st.code("pytest tests/ -v --cov=src", language="bash")

    with st.expander("Train Models"):
        st.code(
            "python -m src.train --data " "data/processed/heart_processed.csv " "--model-dir models", language="bash"
        )

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
    - [FastAPI Docs](http://localhost:8000/docs) -
      Interactive API documentation
    - [GitHub Repository]
      (https://github.com/YOUR_USERNAME/MLOPs_Project)
    - [UCI Heart Disease Dataset]
      (https://archive.ics.uci.edu/ml/datasets/heart+disease)
    """
    )


if __name__ == "__main__":
    main()
