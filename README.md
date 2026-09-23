***# Explainable Diabetes Risk Prediction System***



***An end-to-end machine learning system for diabetes risk prediction using multiple classification models, XGBoost, SHAP-based explainability, and a Streamlit web application.***



***## Features***



***- Data preprocessing and missing-value handling***

***- Invalid zero-value detection***

***- Multiple ML model comparison***

***- Logistic Regression***

***- Support Vector Machine (SVM)***

***- Random Forest***

***- XGBoost***

***- Accuracy, Precision, Recall and F1-Score***

***- ROC-AUC and PR-AUC evaluation***

***- Confusion matrices and performance curves***

***- SHAP-based Explainable AI***

***- Individual prediction explanations***

***- Interactive Streamlit application***



***## Project Structure***



***```text***

***diabetes-risk-prediction/***

***│***

***├── data/***

***│   └── diabetes.csv***

***│***

***├── models/***

***│   ├── logistic\_regression.pkl***

***│   ├── svm.pkl***

***│   ├── random\_forest.pkl***

***│   └── xgboost.pkl***

***│***

***├── reports/***

***│   ├── model\_comparison.csv***

***│   ├── confusion matrices***

***│   ├── ROC curves***

***│   ├── Precision-Recall curves***

***│   └── SHAP explanations***

***│***

***├── src/***

***│   ├── config.py***

***│   ├── preprocessing.py***

***│   ├── train.py***

***│   ├── evaluate.py***

***│   ├── explain.py***

***│   └── predict.py***

***│***

***├── app.py***

***├── requirements.txt***

***├── .gitignore***

***└── README.md***

