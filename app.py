import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import shap
import joblib

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

sys.path.append(str(SRC_DIR))

from predict import predict_diabetes

# --------------------------------------------------
# SHAP LOCAL EXPLANATION
# --------------------------------------------------

def get_shap_explanation(input_values):

    model_path = BASE_DIR / "models" / "xgboost.pkl"

    model_data = joblib.load(model_path)

    model = model_data["model"]
    preprocessor = model_data["preprocessor"]

    # Feature names
    feature_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    # Create dataframe
    input_df = pd.DataFrame(
        [input_values],
        columns=feature_names
    )

    # Replace invalid zeros with missing values
    zero_as_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for column in zero_as_missing:

        if input_df[column].iloc[0] == 0:

            input_df[column] = input_df[column].replace(
                0,
                float("nan")
            )

    # Apply same preprocessing
    input_processed = preprocessor.transform(
        input_df
    )

    input_processed = pd.DataFrame(
        input_processed,
        columns=feature_names
    )

    # Use a small background dataset
    X_background = preprocessor.transform(
        pd.DataFrame(
            {
                "Pregnancies": [1, 2, 3, 4, 5],
                "Glucose": [100, 110, 120, 130, 140],
                "BloodPressure": [65, 70, 72, 75, 80],
                "SkinThickness": [20, 25, 28, 30, 32],
                "Insulin": [80, 90, 100, 110, 120],
                "BMI": [22, 25, 28, 30, 32],
                "DiabetesPedigreeFunction": [
                    0.2, 0.3, 0.4, 0.5, 0.6
                ],
                "Age": [25, 30, 35, 40, 45]
            }
        )
    )

    X_background = pd.DataFrame(
        X_background,
        columns=feature_names
    )

    # SHAP explainer
    explainer = shap.Explainer(
        model.predict_proba,
        X_background
    )

    shap_result = explainer(
        input_processed
    )

    # Get class 1 = diabetes
    if len(shap_result.shape) == 3:

        values = shap_result.values[0, :, 1]

    else:

        values = shap_result.values[0]

    # Create result dataframe
    explanation_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP": values,
            "Value": input_values
        }
    )

    # Sort by absolute impact
    explanation_df["Absolute_SHAP"] = (
        explanation_df["SHAP"].abs()
    )

    explanation_df = explanation_df.sort_values(
        "Absolute_SHAP",
        ascending=False
    )

    return explanation_df

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)



st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .risk-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
    }

    .metric-title {
        font-size: 18px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title"> Diabetes Risk Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Explainable Machine Learning Based Diabetes Risk Assessment
    </div>
    """,
    unsafe_allow_html=True
)


st.info(
    "This application uses a machine learning model to estimate "
    "diabetes risk from the provided health measurements."
)


st.sidebar.title("About the Project")

st.sidebar.write(
    """
    This system uses machine learning to estimate the probability
    of diabetes based on eight input features.

    **Model:** XGBoost

    **Explainability:** SHAP

    **Preprocessing:**
    - Missing-value handling
    - Median imputation
    - Standardization
    """
)

st.sidebar.markdown("---")

st.sidebar.warning(
    "This application is intended for educational and research "
    "purposes only. It is not a medical diagnostic tool."
)



st.header("Patient Information")

st.write(
    "Enter the following health measurements:"
)


col1, col2 = st.columns(2)


with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )



with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=28.5,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.35,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=35,
        step=1
    )



st.markdown("---")

predict_button = st.button(
    "Predict Diabetes Risk",
    use_container_width=True
)



if predict_button:

    try:

        result = predict_diabetes(
            pregnancies=pregnancies,
            glucose=glucose,
            blood_pressure=blood_pressure,
            skin_thickness=skin_thickness,
            insulin=insulin,
            bmi=bmi,
            diabetes_pedigree=diabetes_pedigree,
            age=age
        )

        probability = float(result["probability_percentage"])

        risk_category = result["risk_category"]

        prediction = result["prediction"]


        st.header("Prediction Result")


        result_col1, result_col2, result_col3 = st.columns(3)


        with result_col1:

            st.metric(
                "Estimated Probability",
                f"{probability:.2f}%"
            )


        with result_col2:

            st.metric(
                "Risk Category",
                risk_category
            )


        with result_col3:

            if prediction == 1:

                prediction_text = "Higher Risk"

            else:

                prediction_text = "Lower Risk"

            st.metric(
                "Model Prediction",
                prediction_text
            )


        
        st.subheader("Risk Probability")

        progress_value = float(probability) / 100.0

        st.progress(progress_value)


        
        if risk_category == "Low":

            st.success(
                f"""
                **Model-estimated probability: {probability:.2f}%**

                The model places this input in the **Low**
                probability category.
                """
            )

        elif risk_category == "Moderate":

            st.warning(
                f"""
                **Model-estimated probability: {probability:.2f}%**

                The model places this input in the **Moderate**
                probability category.
                """
            )

        else:

            st.error(
                f"""
                **Model-estimated probability: {probability:.2f}%**

                The model places this input in the **High**
                probability category.
                """
            )


        
        # --------------------------------------------------
        # WHY THIS PREDICTION?
        # --------------------------------------------------
        
        st.subheader("Why this prediction?")
        
        st.write(
            "The explanation below shows which features had the "
            "largest influence on this specific model prediction."
        )
        
        input_values = [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]
        
        try:
        
            explanation_df = get_shap_explanation(
                input_values
            )
        
            # --------------------------------------------------
            # TOP FEATURES
            # --------------------------------------------------
        
            top_features = explanation_df.head(5).copy()
        
            max_impact = top_features["Absolute_SHAP"].max()
        
            display_names = {
                "Pregnancies": "Pregnancies",
                "Glucose": "Glucose",
                "BloodPressure": "Blood Pressure",
                "SkinThickness": "Skin Thickness",
                "Insulin": "Insulin",
                "BMI": "BMI",
                "DiabetesPedigreeFunction":
                    "Diabetes Pedigree Function",
                "Age": "Age"
            }
        
            # --------------------------------------------------
            # TEXT EXPLANATION
            # --------------------------------------------------
        
            for _, row in top_features.iterrows():
        
                feature = row["Feature"]
        
                shap_value = float(row["SHAP"])
        
                display_name = display_names[feature]
        
                if max_impact > 0:
        
                    bar_length = int(
                        (abs(shap_value) / max_impact) * 12
                    )
        
                else:
        
                    bar_length = 1
        
                bar_length = max(
                    1,
                    min(bar_length, 12)
                )
        
                bar = "█" * bar_length
        
                if shap_value > 0:
        
                    direction = "↑"
                    explanation = "increases"
        
                else:
        
                    direction = "↓"
                    explanation = "decreases"
        
                st.markdown(
                    f"**{display_name:<28}** "
                    f"`{bar:<12}` **{direction}**"
                )
        
                st.caption(
                    f"{display_name} {explanation} "
                    f"the model's estimated probability."
                )
        
        
            # --------------------------------------------------
            # SHAP BAR CHART
            # --------------------------------------------------
        
            st.markdown("---")
        
            st.write("### SHAP Feature Contribution")
        
            chart_df = top_features.copy()
        
            chart_df["Feature"] = chart_df["Feature"].map(
                display_names
            )
        
            # Sort for horizontal bar chart
            chart_df = chart_df.sort_values(
                "SHAP"
            )
        
            st.bar_chart(
                chart_df.set_index("Feature")["SHAP"]
            )
        
            st.caption(
                "Positive SHAP values indicate that the feature "
                "pushed the model toward a higher diabetes "
                "probability. Negative values indicate the "
                "opposite for this prediction."
            )
        
        
        except Exception as e:
        
            st.warning(
                f"SHAP explanation could not be generated: {e}"
            )

        
        st.markdown("---")

        st.warning(
            """
            **Important:** This prediction is generated by a machine
            learning model and should not be interpreted as a medical
            diagnosis. Please consult a qualified healthcare
            professional for medical advice or diagnosis.
            """
        )


    except Exception as e:

        st.error(
            f"An error occurred while making the prediction: {e}"
        )