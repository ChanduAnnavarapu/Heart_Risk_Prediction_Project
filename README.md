# 🩺 **Heart Disease Risk Prediction System**
An end-to-end Machine Learning application that predicts the risk of heart disease based on patient medical parameters using a trained Random Forest Classifier model.

## 📌 Problem Statement

Heart disease is one of the leading causes of death worldwide. Early risk prediction can help doctors and patients take preventive measures before severe complications occur.

This project predicts whether a patient is at risk of heart disease using clinical and medical attributes.

## 🧠 Machine Learning Workflow
```
Data Collection
      ↓
EDA & Visualization
      ↓
Data Preprocessing
      ↓
Train-Test Split using GroupShuffleSplit
      ↓
Model Training
      ↓
Hyperparameter Tuning
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
Model Saving (.joblib)
      ↓
FastAPI Backend
      ↓
Streamlit Frontend
      ↓
AWS EC2 Deployment
```
## 📂 Project Structure
```
Heart_Risk_Prediction_Project/
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── predict.py
│   └── training.py
│
├── dataset/
│
├── frontend/
│   └── app.py
│
├── logs/
│   └── app.log
│
├── models/
│   └── heart_disease_model.joblib
│
├── notebook_files/
│   └── 17_1_Meddybuddy_heart_disease.ipynb
│
├── .env
├── .gitignore
├── env_template.txt
├── logger_config.py
├── README.md
└── requirements.txt
```

## 🔍 Exploratory Data Analysis (EDA)

Performed detailed data analysis to understand feature distributions and relationships.

✅ Univariate Analysis

Analyzed individual features such as:
- Age distribution
- Cholesterol levels
- Blood pressure trends
- Target variable distribution

✅ Bivariate Analysis

Studied relationships between features and target variable:

- Age vs Heart Disease
- Cholesterol vs Risk
- Chest Pain vs Target
- Sex vs Heart Disease

✅ Multivariate Analysis

Performed advanced analysis using:

- Correlation heatmaps
- Pair plots
- Feature interaction analysis

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Feature encoding
- Data cleaning
- Feature scaling (if required)
- Removing inconsistencies

Used: **GroupShuffleSplit**

- Group shuffle split is used in this preoject to avoid the dataleakage, found out 723 duplicates rows from the dataset and those are reports of same persons taking multiple consulations.

## 🤖 Model Training

Multiple machine learning algorithms were trained and evaluated.

Models experimented with:

- Logistic Regression
- Support Vector Machine
- Random Forest Classifier
- Gradient Boosting
- XGBoost

## 🎯 Hyperparameter Tuning

Performed hyperparameter optimization to improve model performance.

Techniques used: **RandomizedSearchCV**

Tuned parameters included:

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- criterion

## 🏆 Best Model Selection

After comparing multiple models based on evaluation metrics, the:

✅ Random Forest Classifier was selected as the best-performing model.

Selection Criteria: 
High Accuracy,
Better Generalization,
Strong Recall Score,
Stable Performance

The trained model was saved as: **models/heart_disease_model.joblib**
## 📈 Model Evaluation Metrics

The model was evaluated using:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Confusion Matrix
- ROC-AUC Score

## 🔮 Prediction Pipeline

The prediction logic is implemented in: **backend/predict.py**

**Responsibilities:**

- Load trained model
- Accept input features
- Perform preprocessing
- Generate prediction
- Return prediction result

## 🌐 FastAPI Backend

Main backend file: **backend/main.py**

**API Features**

- REST API endpoints
- JSON request handling
- Real-time predictions
- Swagger API documentation
- Backend logging

## 🖥️ Streamlit Frontend

Frontend file: **frontend/app.py**

**Frontend Features**
- User-friendly UI
- Form-based medical inputs
- Instant prediction results
- FastAPI integration
- Real-time interaction

## ☁️ AWS EC2 Deployment

The complete application was deployed on an AWS EC2 Ubuntu instance using SSH remote access from the local machine.

### Deployment Workflow
- Connected to the EC2 instance securely using SSH
- Cloned the complete project repository from GitHub into the EC2 server
- Created and activated a Python virtual environment
- Installed all required dependencies from requirements.txt
- Started the FastAPI backend service to expose prediction endpoints
- Launched the Streamlit frontend application for user interaction
- Configured the application to run on public ports
- Accessed and validated the deployed application directly through the browser using the EC2 public IP