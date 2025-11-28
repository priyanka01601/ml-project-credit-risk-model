🚀 Live Demo

🔗 Streamlit App: https://ml-project-credit-risk-model-dn76s4drdpfzgvjimkw9ny.streamlit.app/

## 📊 Credit Risk Modelling and Streamlit Prediction App

```
This project builds a complete Credit Risk Prediction System using customer, loan, and bureau data.
It includes advanced feature engineering, WOE/IV analysis, model training (Logistic Regression, Random Forest, XGBoost), imbalance handling, hyperparameter tuning, and deployment through a Streamlit-based prediction app.

This repository demonstrates an end-to-end machine learning workflow suitable for data science, machine learning, and fintech roles.
```

### 📁 Repository Structure
```
ml-project-credit-risk-model/
│
├── Artifacts/
│   └── model.joblib                  # Final deployed model
│
├── Data/                             # (Optional) Public or sample datasets
│   ├── customers.csv
│   ├── loans.csv
│   └── bureau_data.csv
│
├── notebooks/
│   └── Credit Risk Model.ipynb   # Full EDA, modelling & evaluation
│
├── main.py                           # Streamlit app
├── prediction_helper.py              # Preprocessing + prediction pipeline
│
├── requirements.txt
├── LICENSE
└── README.md
```

### 🎯 Objective

Predict loan default risk using demographic, loan-level, and bureau history data.
Outputs:
✔ Default / No Default
✔ Default probability

Deployed as an interactive  Streamlit web app.

### 🧠 Workflow Summary

🔹 1. Data Loading & Merging

      Combined customers, loans, and bureau data into a unified modelling dataset.

🔹 2. Cleaning & Preprocessing

        Missing values
        
        Outlier removal
        
        Duplicate checks
        
        Temporal train–test split

🔹 3. EDA

        Feature distributions
        
        Default vs non-default patterns
        
        Correlations, boxplots, KDEs

🔹 4. Feature Engineering
      
      Derived ratios (loan_to_income, delinquent_ratio)
      
      Loan tenure, net disbursement
      
      WOE/IV-based binning and feature selection

🔹 5. Class Imbalance Handling

      SMOTE
      
      Random Under Sampling

🔹 6. Models

      Built and compared:
      
      Logistic Regression
      
      Random Forest
      
      XGBoost
      
      Hyperparameter tuning (RandomizedSearchCV)

      Best model saved as: Artifacts/model.joblib

🔹 7. Evaluation

      Confusion matrix

      ROC-AUC
      
      Precision–Recall

      Decile analysis (industry standard in credit risk)

      Feature importance

🔹 8. Streamlit Deployment

    The app:

    Accepts customer inputs

    Preprocesses using prediction_helper.py

    Returns risk prediction + probability

Run locally:
```
pip install -r requirements.txt
streamlit run main.py
```

### 🧰 Tech Stack

#### Python :
• Pandas • NumPy • Scikit-learn • XGBoost • SMOTE
#### Visualization: 
• Matplotlib • Seaborn
#### Deployment:
• Streamlit • Joblib

### 📄 Model Summary
```
| Item         | Details                                                 |
| ------------ | ------------------------------------------------------- |
| Model        | Logistic Regression / XGBoost                           |
| Target       | Loan Default (0/1)                                      |
| Metrics      | Recall, ROC-AUC, F1, Decile Capture                     |
| Key Features | loan_to_income, delinquent_ratio, tenure, bureau scores |
| Use Case     | Early detection of high-risk borrowers                  |

