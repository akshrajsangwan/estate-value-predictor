# California Estate Value Predictor

A Machine Learning project that predicts California housing prices using demographic, geographic, and housing-related features. The project follows an end-to-end Machine Learning workflow including data preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation.

---

## Project Overview

The goal of this project is to predict the median house value of a district in California using various housing and population features.

This project demonstrates:

- Data preprocessing
- Feature engineering
- Machine Learning pipelines
- Model comparison
- Cross-validation
- Hyperparameter tuning
- Regression modeling using Scikit-Learn

---

## Dataset

The project uses the California Housing Dataset available through Scikit-Learn and follows concepts presented in:

**Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow**
by Aurélien Géron

### Dataset Information

- Number of records: ~20,640
- Target Variable: `median_house_value`

### Features

| Feature | Description |
|----------|------------|
| longitude | Geographic longitude |
| latitude | Geographic latitude |
| housing_median_age | Median age of houses |
| total_rooms | Total number of rooms |
| total_bedrooms | Total number of bedrooms |
| population | Population of the district |
| households | Number of households |
| median_income | Median income |
| ocean_proximity | Distance from ocean |
| median_house_value | Target variable |

---

## Data Preprocessing

The following preprocessing techniques were applied:

- Missing value handling using `SimpleImputer`
- Feature scaling using `StandardScaler`
- One-Hot Encoding for categorical features
- Automated preprocessing using `ColumnTransformer`
- End-to-end preprocessing using Scikit-Learn Pipelines

---

## Feature Engineering

Additional features were created to improve model performance:

- Bedrooms per Room
- Rooms per Household
- People per Household
- Log Transformation of skewed features
- Geographic Cluster Similarity Features

---

## Train-Test Split

The dataset was split using:

```python
StratifiedShuffleSplit()
```

This ensures that the training and testing datasets maintain similar income category distributions.

---

## Models Evaluated

### 1. Linear Regression

Used as a baseline regression model.

### 2. Decision Tree Regressor

Provides non-linear learning capability but showed signs of overfitting.

### 3. Random Forest Regressor

Delivered the best overall performance and was selected as the final model.

---

## Model Evaluation

Evaluation Metrics:

- Root Mean Squared Error (RMSE)
- 10-Fold Cross Validation

### Results

| Model | RMSE |
|---------|---------|
| Linear Regression | Higher |
| Decision Tree Regressor | ~66,574 |
| Random Forest Regressor | ~47,038 |

---

## Hyperparameter Tuning

The final model was optimized using:

- RandomizedSearchCV
- GridSearchCV

### Best Parameters

```python
{
    'preprocessing__geo__n_clusters': 48,
    'random_forest__max_features': 10
}
```

### Best Cross-Validation RMSE

```text
42,126
```

---

## Machine Learning Pipeline

The pipeline includes:

1. Missing Value Imputation
2. Feature Engineering
3. Feature Scaling
4. Categorical Encoding
5. Model Training

This ensures consistent preprocessing during both training and prediction.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Jupyter Notebook

---

## Project Structure

```text
California-House-Price-Prediction/
│
├── California_House_Price_Prediction.ipynb
├── datasets/
├── README.md
└── requirements.txt
```

---

## Future Improvements

- Save trained model using Joblib
- Build Streamlit Web Application
- Deploy application on Streamlit Cloud
- Add user input interface for real-time predictions

---

## Author

Developed by Akshraj Sangwan

B.Tech CSE (AI & ML)

Machine Learning Project based on concepts from Hands-On Machine Learning by Aurélien Géron.
