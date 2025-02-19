# Backpack Price Prediction 

## 📌 Project Overview
This project aims to predict the prices of backpacks based on various features such as brand, material, size, compartments, waterproof capability, and weight capacity. Using machine learning techniques, we preprocess the data, engineer features, and build predictive models to estimate backpack prices accurately.

## 🛠 Skills & Tools Used
- **Programming:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Feature Engineering:** Label Encoding, One-Hot Encoding, Scaling (StandardScaler)
- **Machine Learning Models:** Linear Regression, Random Forest, XGBoost, Gradient Boosting
- **Hyperparameter Tuning:** GridSearchCV
- **Model Evaluation:** RMSE, MAE, R²
- **Libraries Used:** Scikit-Learn, XGBoost

## 📂 Dataset
The project utilizes two datasets:
- `data_train.csv`: Primary training dataset
- `data_train_extra.csv`: Additional training data

### Features in the Dataset:
- `Brand` - Brand of the backpack
- `Material` - Material type (e.g., leather, polyester)
- `Size` - Backpack size (Small, Medium, Large)
- `Compartments` - Number of compartments
- `Laptop Compartment` - Whether a laptop compartment is present (Yes/No)
- `Waterproof` - Waterproof feature (Yes/No)
- `Style` - Type/style of backpack
- `Color` - Backpack color
- `Weight Capacity (kg)` - Maximum weight capacity
- `Price` - Target variable (price of the backpack)

## 🚀 Project Workflow
### 1️⃣ Exploratory Data Analysis (EDA)
- Checked for missing values and outliers
- Visualized distributions of categorical and numerical features
- Analyzed correlations between features and price

### 2️⃣ Data Preprocessing & Feature Engineering
- Filled missing values (median for numerical, mode for categorical)
- Encoded categorical variables:
  - Label Encoding for `Size`
  - One-Hot Encoding for `Brand`, `Material`, `Style`, and `Color`
- Mapped binary features (`Laptop Compartment`, `Waterproof`) to numerical values (1/0)
- Scaled numerical features using `StandardScaler`

### 3️⃣ Model Development
- Built baseline models:
  - Linear Regression
  - Random Forest Regressor
- Trained advanced models:
  - XGBoost Regressor
  - Gradient Boosting Regressor
- Tuned hyperparameters using `GridSearchCV`

### 4️⃣ Model Evaluation
- Used metrics like RMSE, MAE, and R² to assess model performance
- Compared different models to select the best-performing one

### 5️⃣ Submission Preparation
- Generated predictions on the test dataset
- Prepared a CSV file for submission to the Kaggle competition

## 📊 Results
- Identified that **material and brand significantly impact price**
- **Random Forest and XGBoost performed the best**, with optimized hyperparameters improving accuracy
- **Final model selection based on RMSE and R² scores**

## 📌 Future Improvements
- Experiment with more feature engineering techniques
- Implement deep learning models (Neural Networks)
- Optimize feature selection to reduce noise

## 📁 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/backpack-price-prediction.git
   cd backpack-price-prediction
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook/script for preprocessing and model training.
4. Evaluate results and generate predictions.

