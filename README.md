# Health Insurance Cross-Sell Prediction

This repository is dedicated to solving the **Health Insurance Cross-Sell Prediction** problem using the dataset from Kaggle. The dataset can be found [here](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction). 

## Project Overview

The goal of this project is to predict whether an existing insurance customer will buy vehicle insurance for the next year, based on various features like demographics and previous insurance details.

This dataset provides a variety of features related to the insured person, such as:

- **Age**
- **Gender**
- **Region Code**
- **Policy Sales Channel**
- **Driving License**
- **Vehicle Age**
- **Annual Premium** 

I'll aim to develop a Machine Learning model that can predict the target variable `Response` (1: Will buy insurance, 0: Will not buy insurance).

## Dataset

The dataset consists of one CSV files:

1. `dataset.csv`: The full dataset with features related to health insured persons.

The target variable in the training data is `Response`, which indicates whether the customer will purchase insurance.

| Column Name          | Description                                              |
|----------------------|----------------------------------------------------------|
| `id`                 | Unique identifier for each customer                      |
| `Gender`             | Gender of the customer                                   |
| `Age`                | Age of the customer                                      |
| `Driving_License`    | 0: Customer does not have DL, 1: Customer has DL         |
| `Region_Code`        | Unique code for the region of the customer                |
| `Previously_Insured` | 0: Customer does not have insurance, 1: Customer has insurance |
| `Vehicle_Age`        | Age of the customer’s vehicle                            |
| `Vehicle_Damage`     | 1: Customer has damaged the vehicle, 0: Customer has not damaged the vehicle |
| `Annual_Premium`     | The premium amount for insurance                         |
| `Policy_Sales_Channel`| Channel through which the policy was sold               |
| `Vintage`            | Number of days the customer has been associated with the company |
| `Response`           | 1: Will buy insurance, 0: Will not buy insurance         |

## Approach

### 1. Data Preprocessing
- **Handling Missing Data**: Identify and handle any missing data.
- **Feature Engineering**: Analyze categorical and numerical features to create additional informative features.
- **Normalization/Scaling**: Normalize or scale the numerical features to prepare them for model training.
  
### 2. Exploratory Data Analysis (EDA)
- Perform descriptive statistics and visualization to understand data distributions, correlations, and patterns.
- Analyze class imbalance in the target variable and address it if necessary.

### 3. Model Development
- **Model Selection**: We will experiment with multiple models including:
    - Logistic Regression
    - Decision Trees
    - Random Forests
    - Gradient Boosting Machines (XGBoost, LightGBM)
    - Neural Networks
- **Hyperparameter Tuning**: Optimize model parameters using cross-validation techniques.
- **Evaluation Metrics**: Accuracy, ROC-AUC, F1-Score, Precision, and Recall will be used to evaluate the model's performance.

### 4. Model Deployment
- Finalize the best-performing model and save it for potential deployment.
- Explore model interpretability and SHAP values to understand the features driving predictions.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/HugoTex98/health-insurance-cross-sell-prediction.git
    cd health-insurance-cross-sell-prediction
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction?resource=download&select=train.csv) and place it in the `/dataset` directory.

## Usage

1. Run the preprocessing script to clean and prepare the data:
    ```bash
    python scripts/preprocess.py
    ```

2. Train the model:
    ```bash
    python scripts/train_model.py
    ```

3. Evaluate the model on the test set:
    ```bash
    python scripts/evaluate.py
    ```

## Acknowledgements

- Kaggle for providing the dataset.
