import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from preprocessing import prepare_data
from config import MODEL_DIR, FEATURES

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_xgboost_model():

    model_path = MODEL_DIR / "xgboost.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "XGBoost model not found. Run train.py first."
        )

    model_data = joblib.load(model_path)

    return model_data


def prepare_shap_data():

    X_train, X_test, y_train, y_test = prepare_data()

    model_data = load_xgboost_model()

    model = model_data["model"]
    preprocessor = model_data["preprocessor"]

    
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=FEATURES
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=FEATURES
    )

    return (
        model,
        X_train_processed,
        X_test_processed,
        y_test
    )



def create_global_explanation():

    print("\nCreating global SHAP explanation...")

    model, X_train, X_test, y_test = prepare_shap_data()

    

    background = shap.sample(
        X_train,
        100,
        random_state=42
    )

    print("Creating SHAP explainer...")

    explainer = shap.Explainer(
        model.predict_proba,
        background
    )

    print("Calculating SHAP values...")

    shap_values = explainer(X_test)


    if len(shap_values.shape) == 3:

        diabetes_shap = shap_values[:, :, 1]

    else:

        diabetes_shap = shap_values


   
    print("Creating SHAP summary plot...")

    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        diabetes_shap.values,
        X_test,
        show=False
    )

    plt.title(
        "SHAP Feature Impact on Diabetes Prediction"
    )

    plt.tight_layout()

    output_path = REPORT_DIR / "shap_summary.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


    print("Creating SHAP feature importance plot...")

    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        diabetes_shap.values,
        X_test,
        plot_type="bar",
        show=False
    )

    plt.title(
        "Global Feature Importance - SHAP"
    )

    plt.tight_layout()

    output_path = REPORT_DIR / "shap_feature_importance.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")



if __name__ == "__main__":

    print("=" * 60)
    print("EXPLAINABLE AI - SHAP")
    print("=" * 60)

    create_global_explanation()

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 60)