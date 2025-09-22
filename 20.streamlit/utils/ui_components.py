
import streamlit as st


def show_manual_hyperparameter_ui(model_name):
    """Displays widgets for manual hyperparameter tuning in the sidebar."""
    params = {}
    st.sidebar.markdown("---")
    st.sidebar.header("Manual Hyperparameters")

    if model_name == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("Number of Estimators", 10, 200, 100, 10)
        params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 10, 1)
        params["min_samples_leaf"] = st.sidebar.slider("Min Samples Leaf", 1, 20, 1, 1)

    elif model_name == "Logistic Regression":
        params["C"] = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
        params["solver"] = st.sidebar.selectbox("Solver", ["lbfgs", "newton-cg", "sag", "saga"])

    elif model_name == "Support Vector Machine (SVM)":
        params["C"] = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
        params["kernel"] = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])

    elif model_name == "Gradient Boosting":
        params["n_estimators"] = st.sidebar.slider("Number of Estimators", 50, 500, 100, 10)
        params["learning_rate"] = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
        params["max_depth"] = st.sidebar.slider("Max Depth", 1, 15, 3, 1)

    return params
