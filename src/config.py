from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

DATA_FILE = DATA_DIR / "diabetes.csv"

MODEL_DIR = BASE_DIR / "models"
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


TARGET = "Outcome"


ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]
TEST_SIZE = 0.20

RANDOM_STATE = 42

CV_FOLDS = 5

MODEL_NAMES = [
    "Logistic Regression",
    "SVM",
    "Random Forest",
    "XGBoost"
]