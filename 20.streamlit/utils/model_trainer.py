from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import streamlit as st
import numpy as np
import pandas as pd
import time
from typing import Dict, Any

class ModelTrainer:
    """
    Enhanced model trainer specifically designed for Streamlit applications.
    Provides better user feedback, error handling, and training control.
    """

    @staticmethod
    def get_models():
        """
        Returns a dictionary of initialized model instances optimized for Streamlit.
        Uses n_jobs=1 to avoid multiprocessing conflicts in Streamlit environment.
        """
        return {
            "Random Forest": RandomForestClassifier(
                random_state=42,
                n_jobs=1,
                warm_start=False
            ),
            "Logistic Regression": LogisticRegression(
                random_state=42,
                max_iter=2000,
                n_jobs=1  # !! Multiple threads (jobs) cause conflicts in streamlit
            ),
            "Support Vector Machine (SVM)": SVC(
                random_state=42,
                probability=True,
                cache_size=200
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42,
                warm_start=False
            ),
        }

    @staticmethod
    def get_param_grids(quick_mode=False):
        """
        Returns parameter grids with options for quick vs comprehensive search.

        Args:
            quick_mode: If True, returns smaller grids for faster training
        """
        if quick_mode:
            return {
                "Random Forest": {
                    "classifier__n_estimators": [50, 100],
                    "classifier__max_depth": [10, 20],
                },
                "Logistic Regression": {
                    "classifier__C": [0.1, 1.0, 10.0],
                },
                "Support Vector Machine (SVM)": {
                    "classifier__C": [1.0, 10.0],
                    "classifier__kernel": ["rbf", "linear"],
                },
                "Gradient Boosting": {
                    "classifier__n_estimators": [50, 100],
                    "classifier__learning_rate": [0.1, 0.2],
                },
            }
        else:
            return {
                "Random Forest": {
                    "classifier__n_estimators": [50, 100, 200],
                    "classifier__max_depth": [5, 10, 20, None],
                    "classifier__min_samples_leaf": [1, 2, 4],
                    "classifier__min_samples_split": [2, 5, 10],
                },
                "Logistic Regression": {
                    "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "classifier__solver": ["lbfgs", "liblinear"],
                },
                "Support Vector Machine (SVM)": {
                    "classifier__C": [0.1, 1.0, 10.0, 100.0],
                    "classifier__kernel": ["rbf", "linear", "poly"],
                    "classifier__gamma": ["scale", "auto"],
                },
                "Gradient Boosting": {
                    "classifier__n_estimators": [50, 100, 200],
                    "classifier__learning_rate": [0.01, 0.1, 0.2],
                    "classifier__max_depth": [3, 5, 7],
                    "classifier__subsample": [0.8, 1.0],
                },
            }

    @staticmethod
    def create_pipeline(model, include_scaler=True):
        """
        Creates an enhanced pipeline with optional scaling.

        Args:
            model: An instantiated scikit-learn classifier
            include_scaler: Whether to include StandardScaler (some models don't need it)
        """
        steps = [("imputer", SimpleImputer(strategy="median"))]

        if include_scaler:
            steps.append(("scaler", StandardScaler()))

        steps.append(("classifier", model))

        return Pipeline(steps=steps)

    @staticmethod
    def validate_data(X_train, y_train, X_test=None, y_test=None):
        """
        Validates training data before model training.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if X_train is None or y_train is None:
                return False, "Training data is missing"

            if len(X_train) == 0 or len(y_train) == 0:
                return False, "Training data is empty"

            if X_train.shape[0] != y_train.shape[0]:
                return False, "Feature and target arrays have different lengths"

            if X_train.shape[0] < 10:
                return False, "Need at least 10 samples for reliable training"

            if X_train.shape[1] == 0:
                return False, "No features found in training data"

            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                return False, "Need at least 2 classes for classification"

            class_counts = pd.Series(y_train).value_counts()
            min_class_ratio = class_counts.min() / class_counts.max()
            if min_class_ratio < 0.1:
                st.warning(f"⚠️ Severe class imbalance detected (ratio: {min_class_ratio:.2f}). Consider using stratified sampling or class weighting.")

            return True, "Data validation passed"

        except Exception as e:
            return False, f"Data validation error: {str(e)}"

    @staticmethod
    def train_model_with_progress(
        model_name: str,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        use_grid_search: bool = True,
        quick_mode: bool = False,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Trains a model with Streamlit progress tracking and comprehensive results.

        Args:
            model_name: Name of the model to train
            X_train, y_train: Training data
            X_test, y_test: Optional test data
            use_grid_search: Whether to use hyperparameter tuning
            quick_mode: Use smaller parameter grids for faster training
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary containing training results and model
        """

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("🔍 Validating data...")
            progress_bar.progress(10)

            is_valid, message = ModelTrainer.validate_data(X_train, y_train, X_test, y_test)
            if not is_valid:
                st.error(f"❌ {message}")
                return {"success": False, "error": message}

            status_text.text("🔧 Initializing model...")
            progress_bar.progress(20)

            models = ModelTrainer.get_models()
            if model_name not in models:
                error_msg = f"Model '{model_name}' not found"
                st.error(f"❌ {error_msg}")
                return {"success": False, "error": error_msg}

            model = models[model_name]

            # Determine if scaling is needed (SVM and Logistic Regression benefit from scaling)
            needs_scaling = model_name in ["Support Vector Machine (SVM)", "Logistic Regression"]
            pipeline = ModelTrainer.create_pipeline(model, include_scaler=needs_scaling)

            # Step 3: Training
            start_time = time.time()

            if use_grid_search:
                status_text.text("🔍 Performing hyperparameter tuning...")
                progress_bar.progress(30)

                param_grids = ModelTrainer.get_param_grids(quick_mode=quick_mode)
                param_grid = param_grids[model_name]

                search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=cv_folds,
                    n_jobs=1,
                    verbose=0,
                    scoring='accuracy',
                    return_train_score=True
                )

                search.fit(X_train, y_train)
                best_model = search.best_estimator_

                progress_bar.progress(70)
                status_text.text("✅ Hyperparameter tuning completed!")

            else:
                status_text.text("🚀 Training model with default parameters...")
                progress_bar.progress(40)

                pipeline.fit(X_train, y_train)
                best_model = pipeline
                search = None

                progress_bar.progress(70)

            status_text.text("📊 Evaluating model performance...")
            progress_bar.progress(80)

            train_predictions = best_model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_predictions)

            cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring='accuracy', n_jobs=1)

            test_accuracy = None
            test_predictions = None
            if X_test is not None and y_test is not None:
                test_predictions = best_model.predict(X_test)
                test_accuracy = accuracy_score(y_test, test_predictions)

            training_time = time.time() - start_time

            progress_bar.progress(90)
            status_text.text("📋 Compiling results...")

            results = {
                "success": True,
                "model_name": model_name,
                "model": best_model,
                "training_time": training_time,
                "train_accuracy": train_accuracy,
                "cv_mean_accuracy": cv_scores.mean(),
                "cv_std_accuracy": cv_scores.std(),
                "cv_scores": cv_scores,
                "test_accuracy": test_accuracy,
                "train_predictions": train_predictions,
                "test_predictions": test_predictions,
                "feature_count": X_train.shape[1],
                "training_samples": X_train.shape[0],
                "test_samples": X_test.shape[0] if X_test is not None else 0,
            }

            if search:
                results.update({
                    "best_params": search.best_params_,
                    "best_cv_score": search.best_score_,
                    "grid_search_results": search.cv_results_,
                })

            progress_bar.progress(100)
            status_text.text("✅ Model training completed successfully!")

            st.success(f"""
            🎉 **{model_name} Training Complete!**

            📈 **Key Metrics:**
            - Training Accuracy: {train_accuracy:.4f}
            - Cross-validation: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})
            {f"- Test Accuracy: {test_accuracy:.4f}" if test_accuracy else ""}
            - Training Time: {training_time:.2f} seconds
            {f"- Best Parameters: {search.best_params_}" if search else ""}
            """)

            return results

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            st.error(f"❌ {error_msg}")
            status_text.text("❌ Training failed!")
            progress_bar.progress(0)

            return {
                "success": False,
                "error": error_msg,
                "model_name": model_name
            }

    @staticmethod
    def quick_train_all_models(X_train, y_train, X_test=None, y_test=None) -> Dict[str, Any]:
        """
        Quickly trains all available models for comparison.

        Returns:
            Dictionary with results for each model
        """
        models = ModelTrainer.get_models()
        all_results = {}

        total_models = len(models)
        overall_progress = st.progress(0)
        status_container = st.empty()

        for i, model_name in enumerate(models.keys()):
            with status_container.container():
                st.write(f"🔄 Training {model_name} ({i+1}/{total_models})...")

                result = ModelTrainer.train_model_with_progress(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    use_grid_search=False,
                    quick_mode=True
                )

                all_results[model_name] = result
                overall_progress.progress((i + 1) / total_models)

        status_container.empty()
        overall_progress.progress(1.0)

        if all(result.get("success", False) for result in all_results.values()):
            st.success("🎉 All models trained successfully!")

            comparison_data = []
            for name, result in all_results.items():
                if result.get("success"):
                    comparison_data.append({
                        "Model": name,
                        "CV Accuracy": f"{result['cv_mean_accuracy']:.4f}",
                        "Training Time": f"{result['training_time']:.2f}s",
                        "Test Accuracy": f"{result['test_accuracy']:.4f}" if result.get('test_accuracy') else "N/A"
                    })

            if comparison_data:
                st.write("📊 **Model Comparison:**")
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

        return all_results
