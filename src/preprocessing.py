import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import (
    DATA_FILE,
    FEATURES,
    TARGET,
    ZERO_AS_MISSING,
    TEST_SIZE,
    RANDOM_STATE
)

def load_data():
    
    df = pd.read_csv(r'C:\Users\aravi\Projects\Project\data\diabetes.csv')

    return df

def validate_data(df):
    
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in dataset: {missing_columns}"
        )

    return True

def replace_invalid_zeros(df):
    
    df = df.copy()

    for column in ZERO_AS_MISSING:

        if column in df.columns:
            df[column] = df[column].replace(0, np.nan)

    return df



def split_features_target(df):
    
    X = df[FEATURES]

    y = df[TARGET]

    return X, y


def split_data(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor():
    
    preprocessing_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    return preprocessing_pipeline


def prepare_data():
    
    df = load_data()

    validate_data(df)

    df = replace_invalid_zeros(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    return X_train, X_test, y_train, y_test



if __name__ == "__main__":

    print("=" * 60)
    print("DIABETES DATA PREPROCESSING")
    print("=" * 60)

    X_train, X_test, y_train, y_test = prepare_data()

    print("\nDataset successfully loaded!")

    print("\nTraining data shape:")
    print(X_train.shape)

    print("\nTesting data shape:")
    print(X_test.shape)

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    print("\nCreating preprocessing pipeline...")

    preprocessor = create_preprocessor()

    print("Preprocessing pipeline created successfully!")

    print("\nPipeline steps:")
    print(preprocessor)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)