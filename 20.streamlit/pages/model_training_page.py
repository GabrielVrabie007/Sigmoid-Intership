import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from utils.model_trainer import ModelTrainer as model_trainer


@st.cache_data
def load_data(path):
    """Load CSV data and inject some NaNs for demonstration."""
    df = pd.read_csv(path)
    # Specially inject some NaNs because dataset is too cleaner and small only 178 samples
    for col in df.columns:
        if col != 'Customer_Segment':
            sample_indices = df.sample(frac=0.03, random_state=42).index
            df.loc[sample_indices, col] = np.nan
    return df


def show_manual_hyperparameter_ui(model_name):
    """Enhanced manual hyperparameter UI with better defaults and validation."""
    params = {}
    st.sidebar.markdown("---")
    st.sidebar.header("Manual Hyperparameters")
    st.sidebar.info("💡 Tip: Start with default values and adjust incrementally")

    if model_name == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("Number of Estimators", 10, 500, 100, 10)
        params["max_depth"] = st.sidebar.slider("Max Depth", 3, 30, 10, 1)
        params["min_samples_leaf"] = st.sidebar.slider("Min Samples Leaf", 1, 20, 1, 1)
        params["min_samples_split"] = st.sidebar.slider("Min Samples Split", 2, 20, 2, 1)

    elif model_name == "Logistic Regression":
        params["C"] = st.sidebar.slider("Regularization (C)", 0.001, 100.0, 1.0, step=0.001, format="%.3f")
        params["solver"] = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"])
        params["max_iter"] = st.sidebar.slider("Max Iterations", 100, 5000, 2000, 100)

    elif model_name == "Support Vector Machine (SVM)":
        params["C"] = st.sidebar.slider("Regularization (C)", 0.001, 100.0, 1.0, step=0.001, format="%.3f")
        params["kernel"] = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        if params["kernel"] == "rbf":
            params["gamma"] = st.sidebar.selectbox("Gamma", ["scale", "auto"])

    elif model_name == "Gradient Boosting":
        params["n_estimators"] = st.sidebar.slider("Number of Estimators", 50, 500, 100, 10)
        params["learning_rate"] = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1, 0.001, format="%.3f")
        params["max_depth"] = st.sidebar.slider("Max Depth", 1, 15, 3, 1)
        params["subsample"] = st.sidebar.slider("Subsample", 0.5, 1.0, 1.0, 0.1)

    return params


def plot_classification_metrics(y_test, y_pred, class_names):
    """Enhanced bar plot for per-class precision, recall, F1-score."""
    try:
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose()

        # Filter for class metrics only
        class_names_str = [str(c) for c in class_names]
        class_metrics = report_df.loc[report_df.index.intersection(class_names_str),
                                     ['precision', 'recall', 'f1-score']]

        if not class_metrics.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            class_metrics.plot(kind='bar', ax=ax, width=0.8)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Score")
            ax.set_title("Per-Class Performance Metrics")
            ax.legend(loc='lower right')
            ax.set_xlabel("Classes")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Could not generate per-class metrics plot.")

    except Exception as e:
        st.error(f"Error generating classification metrics plot: {str(e)}")


def plot_confusion_matrix_heatmap(y_test, y_pred, class_names):
    """Enhanced confusion matrix heatmap with better formatting."""
    try:
        cm = confusion_matrix(y_test, y_pred)

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        ax1.set_title("Confusion Matrix (Counts)")

        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        ax2.set_title("Confusion Matrix (Percentages)")

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error generating confusion matrix: {str(e)}")


def display_training_results(results):
    """Display comprehensive training results."""
    if not results.get("success", False):
        st.error(f"❌ Training failed: {results.get('error', 'Unknown error')}")
        return None

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🎯 Test Accuracy",
            value=f"{results.get('test_accuracy', 0):.4f}" if results.get('test_accuracy') else "N/A"
        )

    with col2:
        st.metric(
            label="📊 CV Score",
            value=f"{results.get('cv_mean_accuracy', 0):.2%}",
            delta=f"±{results.get('cv_std_accuracy', 0):.3f}"
        )

    with col3:
        st.metric(
            label="⏱️ Training Time",
            value=f"{results.get('training_time', 0):.1f}s"
        )

    with col4:
        st.metric(
            label="📝 Features",
            value=f"{results.get('feature_count', 0)}"
        )

    if results.get('best_params'):
        st.markdown("#### 🔧 Best Hyperparameters Found:")
        clean_params = {k.replace('classifier__', ''): v for k, v in results['best_params'].items()}
        st.json(clean_params)

    return results['model']


def show_feature_importance(model, feature_names):
    """Display feature importance if available."""
    try:
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['classifier']
        else:
            classifier = model

        if hasattr(classifier, 'feature_importances_'):
            st.markdown("---")
            st.markdown("#### 📈 Feature Importance")

            importances_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': classifier.feature_importances_
            }).sort_values('Importance', ascending=False)

            col1, col2 = st.columns([2, 1])

            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                top_features = importances_df.head(10)
                ax.barh(range(len(top_features)), top_features['Importance'])
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['Feature'])
                ax.set_xlabel('Importance')
                ax.set_title('Top 10 Feature Importance')
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                st.markdown("**Top 5 Features:**")
                for i, (_, row) in enumerate(importances_df.head(5).iterrows(), 1):
                    st.write(f"{i}. **{row['Feature']}**: {row['Importance']:.4f}")

        elif hasattr(classifier, 'coef_'):
            st.markdown("---")
            st.markdown("#### 📈 Feature Coefficients")

            coef = classifier.coef_[0] if len(classifier.coef_.shape) > 1 else classifier.coef_
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coef
            })
            coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
            coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['red' if x < 0 else 'blue' for x in coef_df.head(10)['Coefficient']]
            ax.barh(range(len(coef_df.head(10))), coef_df.head(10)['Coefficient'], color=colors)
            ax.set_yticks(range(len(coef_df.head(10))))
            ax.set_yticklabels(coef_df.head(10)['Feature'])
            ax.set_xlabel('Coefficient Value')
            ax.set_title('Top 10 Feature Coefficients')
            ax.invert_yaxis()
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.warning(f"Could not display feature importance: {str(e)}")


def show_model_training_page():
    st.title("🍇 Wine Classifier Training")
    st.write("Train a classifier using best practices with enhanced data preprocessing and comprehensive hyperparameter tuning.")

    try:
        df = load_data("data/Wine.csv")
        st.success(f"✅ Data loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
    except FileNotFoundError:
        st.error("❌ Error: 'data/Wine.csv' not found. Please ensure the file exists in the data directory.")
        st.info("💡 Expected file structure: `data/Wine.csv`")
        return
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return

    if 'Customer_Segment' not in df.columns:
        st.error("❌ Target column 'Customer_Segment' not found in the dataset.")
        return

    X = df.drop(columns=['Customer_Segment'])
    y = df['Customer_Segment']

    with st.expander("📊 Dataset Overview", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(X.columns))
        with col3:
            st.metric("Classes", len(y.unique()))

        st.write("**Class Distribution:**")
        class_dist = y.value_counts().sort_index()
        st.bar_chart(class_dist)

        missing_info = X.isnull().sum()
        if missing_info.sum() > 0:
            st.warning(f"⚠️ Found {missing_info.sum()} missing values across {(missing_info > 0).sum()} features")
            st.write("Missing values will be handled automatically by the preprocessing pipeline.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        class_names = sorted(y.unique())

        st.info(f"📊 Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    except Exception as e:
        st.error(f"❌ Error in train-test split: {str(e)}")
        return

    st.sidebar.header("🔧 Classifier Configuration")

    model_name = st.sidebar.selectbox(
        "Choose a Model",
        ["Random Forest", "Logistic Regression", "Support Vector Machine (SVM)", "Gradient Boosting"],
        help="Select the machine learning algorithm to train"
    )

    tuning_method = st.sidebar.radio(
        "Select Hyperparameter Tuning Method",
        ('Automatic (Grid Search)', 'Manual (Custom Parameters)', 'Quick Train (Default Params)'),
        help="**Automatic:** Uses GridSearchCV for optimal parameters\n**Manual:** Set custom parameters\n**Quick:** Fast training with defaults"
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🚀 Training Options")

    if tuning_method == 'Automatic (Grid Search)':
        quick_mode = st.sidebar.checkbox(
            "Quick Mode",
            value=False,
            help="Use smaller parameter grids for faster training"
        )
        cv_folds = st.sidebar.slider("CV Folds", 3, 10, 5, help="Number of cross-validation folds")
    else:
        quick_mode = True
        cv_folds = 5

    manual_params = {}
    if tuning_method == 'Manual (Custom Parameters)':
        manual_params = show_manual_hyperparameter_ui(model_name)

    st.markdown("---")
    st.subheader(f"🚀 Training: {model_name}")

    train_button_text = {
        'Automatic (Grid Search)': f"🔍 Train with Grid Search",
        'Manual (Custom Parameters)': f"⚙️ Train with Custom Parameters",
        'Quick Train (Default Params)': f"⚡ Quick Train"
    }[tuning_method]

    if st.button(train_button_text, type="primary", use_container_width=True):

        use_grid_search = (tuning_method == 'Automatic (Grid Search)')

        if tuning_method == 'Manual (Custom Parameters)':
            st.info("🔧 Training model with custom hyperparameters...")

            try:
                models = model_trainer.get_models()
                model = models[model_name]

                for param, value in manual_params.items():
                    setattr(model, param, value)

                pipeline = model_trainer.create_pipeline(model)

                results = model_trainer.train_model_with_progress(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    use_grid_search=False,
                    quick_mode=True,
                    cv_folds=cv_folds
                )

            except Exception as e:
                st.error(f"❌ Manual training error: {str(e)}")
                return

        else:
            results = model_trainer.train_model_with_progress(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                use_grid_search=use_grid_search,
                quick_mode=quick_mode,
                cv_folds=cv_folds
            )

        trained_model = display_training_results(results)

        if trained_model and results.get("success"):
            y_pred = results.get('test_predictions')
            if y_pred is not None:

                st.markdown("---")
                st.subheader("📊 Model Performance Visualization")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Confusion Matrix")
                    plot_confusion_matrix_heatmap(y_test, y_pred, class_names)

                with col2:
                    st.markdown("#### Per-Class Metrics")
                    plot_classification_metrics(y_test, y_pred, class_names)

                show_feature_importance(trained_model, X.columns.tolist())

                st.markdown("---")
                st.subheader("🔍 Model Insights")

                with st.expander("📋 Detailed Classification Report", expanded=False):
                    report = classification_report(y_test, y_pred, target_names=[f"Class {c}" for c in class_names])
                    st.text(report)

                if results.get('cv_scores') is not None:
                    with st.expander("📈 Cross-Validation Scores", expanded=False):
                        cv_scores = results['cv_scores']
                        cv_df = pd.DataFrame({
                            'Fold': range(1, len(cv_scores) + 1),
                            'Accuracy': cv_scores
                        })
                        st.bar_chart(cv_df.set_index('Fold'))
                        st.write(f"**Mean CV Score:** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    st.markdown("---")
    st.subheader("⚡ Quick Model Comparison")
    if st.button("🏆 Compare All Models", help="Train all models quickly for comparison"):
        all_results = model_trainer.quick_train_all_models(X_train, y_train, X_test, y_test)

        if all_results:
            comparison_data = []
            for name, result in all_results.items():
                if result.get("success"):
                    comparison_data.append({
                        "Model": name,
                        "Test Accuracy": f"{result.get('test_accuracy', 0):.4f}",
                        "CV Mean": f"{result.get('cv_mean_accuracy', 0):.4f}",
                        "CV Std": f"{result.get('cv_std_accuracy', 0):.4f}",
                        "Training Time (s)": f"{result.get('training_time', 0):.2f}",
                    })

            if comparison_data:
                st.markdown("#### 🏆 Model Comparison Results")
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)

                best_model_idx = comparison_df['Test Accuracy'].astype(float).idxmax()
                best_model_name = comparison_df.iloc[best_model_idx]['Model']
                best_accuracy = comparison_df.iloc[best_model_idx]['Test Accuracy']

                st.success(f"🥇 **Best Model:** {best_model_name} with {best_accuracy} test accuracy!")
