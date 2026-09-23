# predict.py

import joblib
import pandas as pd

from pathlib import Path

from config import MODEL_DIR, FEATURES



def load_model():

    model_path = MODEL_DIR / "xgboost.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "XGBoost model not found. "
            "Please run train.py first."
        )

    model_data = joblib.load(model_path)

    return model_data

def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
):

    model_data = load_model()

    model = model_data["model"]
    preprocessor = model_data["preprocessor"]

   
    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=FEATURES
    )

    zero_as_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for column in zero_as_missing:

        if input_data[column].iloc[0] == 0:
            input_data[column] = input_data[column].replace(
                0,
                float("nan")
            )

    
    input_processed = preprocessor.transform(
        input_data
    )

    
    prediction = model.predict(
        input_processed
    )[0]

    probability = model.predict_proba(
        input_processed
    )[0][1]

    probability_percentage = probability * 100

    
    if probability < 0.30:

        risk_category = "Low"

    elif probability < 0.60:

        risk_category = "Moderate"

    else:

        risk_category = "High"

    return {
        "prediction": int(prediction),
        "probability": probability,
        "probability_percentage": probability_percentage,
        "risk_category": risk_category
    }



if __name__ == "__main__":

    print("=" * 60)
    print("DIABETES RISK PREDICTION")
    print("=" * 60)

    result = predict_diabetes(
        pregnancies=2,
        glucose=120,
        blood_pressure=70,
        skin_thickness=25,
        insulin=80,
        bmi=28.5,
        diabetes_pedigree=0.35,
        age=35
    )

    print("\nPrediction Result")
    print("-" * 60)

    print(
        f"Prediction          : {result['prediction']}"
    )

    print(
        f"Estimated Probability: "
        f"{result['probability_percentage']:.2f}%"
    )

    print(
        f"Risk Category       : "
        f"{result['risk_category']}"
    )

    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)