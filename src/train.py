import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from preprocessing import prepare_data, create_preprocessor

from config import MODEL_DIR, RANDOM_STATE

def create_models():
    
    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

        "SVM": SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            base_score=0.5
        )
    }

    return models


def train_models():

    print("=" * 60)
    print("DIABETES RISK PREDICTION - MODEL TRAINING")
    print("=" * 60)

    X_train, X_test, y_train, y_test = prepare_data()

    print("\nData loaded successfully.")

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    
    preprocessor = create_preprocessor()

    models = create_models()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    
    for model_name, model in models.items():

        print("\n" + "-" * 60)
        print(f"Training: {model_name}")
        print("-" * 60)

        X_train_processed = preprocessor.fit_transform(X_train)

        X_test_processed = preprocessor.transform(X_test)

        model.fit(
            X_train_processed,
            y_train
        )

        model_filename = (
            model_name.lower()
            .replace(" ", "_")
            + ".pkl"
        )

        model_path = MODEL_DIR / model_filename

        joblib.dump(
            {
                "model": model,
                "preprocessor": preprocessor,
                "features": list(X_train.columns)
            },
            model_path
        )

        print(f"{model_name} trained successfully.")
        print(f"Saved to: {model_path}")

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    train_models()