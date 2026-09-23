import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve
)

from preprocessing import prepare_data

from config import MODEL_DIR

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)



def load_models():

    models = {}

    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "SVM": "svm.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl"
    }

    for model_name, filename in model_files.items():

        model_path = MODEL_DIR / filename

        if not model_path.exists():
            print(f"Warning: {filename} not found.")
            continue

        models[model_name] = joblib.load(model_path)

    return models



def evaluate_model(model_name, model_data, X_test, y_test):

    model = model_data["model"]
    preprocessor = model_data["preprocessor"]


    X_test_processed = preprocessor.transform(X_test)

    
    y_pred = model.predict(X_test_processed)

    
    if hasattr(model, "predict_proba"):

        y_probability = model.predict_proba(
            X_test_processed
        )[:, 1]

    else:

        y_probability = model.decision_function(
            X_test_processed
        )

    
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability
    )

   
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Confusion Matrix": cm,
        "y_pred": y_pred,
        "y_probability": y_probability
    }


def save_comparison(results):

    comparison_data = []

    for result in results:

        comparison_data.append({
            "Model": result["Model"],
            "Accuracy": result["Accuracy"],
            "Precision": result["Precision"],
            "Recall": result["Recall"],
            "F1 Score": result["F1 Score"],
            "ROC-AUC": result["ROC-AUC"],
            "PR-AUC": result["PR-AUC"]
        })

    comparison_df = pd.DataFrame(
        comparison_data
    )

    comparison_path = REPORT_DIR / "model_comparison.csv"

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    return comparison_df



def plot_confusion_matrices(results):

    for result in results:

        cm = result["Confusion Matrix"]

        model_name = result["Model"]

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Non-Diabetic",
                "Diabetic"
            ]
        )

        display.plot()

        plt.title(
            f"Confusion Matrix - {model_name}"
        )

        filename = (
            model_name.lower()
            .replace(" ", "_")
            + "_confusion_matrix.png"
        )

        plt.savefig(
            REPORT_DIR / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


def plot_roc_curves(results, y_test):

    plt.figure(figsize=(8, 6))

    for result in results:

        fpr, tpr, _ = roc_curve(
            y_test,
            result["y_probability"]
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f'{result["Model"]} '
                f'(AUC = {result["ROC-AUC"]:.3f})'
            )
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve Comparison")

    plt.legend()

    plt.grid()

    plt.savefig(
        REPORT_DIR / "roc_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_precision_recall_curves(results, y_test):

    plt.figure(figsize=(8, 6))

    for result in results:

        precision, recall, _ = precision_recall_curve(
            y_test,
            result["y_probability"]
        )

        plt.plot(
            recall,
            precision,
            label=(
                f'{result["Model"]} '
                f'(AP = {result["PR-AUC"]:.3f})'
            )
        )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve Comparison")

    plt.legend()

    plt.grid()

    plt.savefig(
        REPORT_DIR / "precision_recall_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def evaluate_all_models():

    print("=" * 60)
    print("DIABETES MODEL EVALUATION")
    print("=" * 60)

    
    X_train, X_test, y_train, y_test = prepare_data()

    models = load_models()

    if not models:

        print("\nNo trained models found.")

        print("Run train.py first.")

        return

    
    results = []

    for model_name, model_data in models.items():

        print("\n" + "-" * 60)

        print(f"Evaluating: {model_name}")

        print("-" * 60)

        result = evaluate_model(
            model_name,
            model_data,
            X_test,
            y_test
        )

        results.append(result)

        print(
            f"Accuracy : {result['Accuracy']:.4f}"
        )

        print(
            f"Precision: {result['Precision']:.4f}"
        )

        print(
            f"Recall   : {result['Recall']:.4f}"
        )

        print(
            f"F1 Score : {result['F1 Score']:.4f}"
        )

        print(
            f"ROC-AUC  : {result['ROC-AUC']:.4f}"
        )

        print(
            f"PR-AUC   : {result['PR-AUC']:.4f}"
        )

    
    comparison_df = save_comparison(
        results
    )

    
    print("\n")
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        comparison_df.round(4).to_string(
            index=False
        )
    )

    
    print("\nCreating confusion matrices...")

    plot_confusion_matrices(
        results
    )

    print("Creating ROC curves...")

    plot_roc_curves(
        results,
        y_test
    )

    print("Creating Precision-Recall curves...")

    plot_precision_recall_curves(
        results,
        y_test
    )

   
    print("\n" + "=" * 60)

    print("EVALUATION COMPLETE")

    print("=" * 60)

    print(f"\nReports saved in:")

    print(REPORT_DIR)


if __name__ == "__main__":

    evaluate_all_models()