# Laptop Price Prediction

Predicts the price of a laptop from its specifications using supervised regression.

**Live app:**  https://laptop-price-poojapradeep.streamlit.app/

## Problem
Given a laptop's brand, type, RAM, screen, processor, storage, graphics card and operating
system, predict its price.

## Data
1,303 rows and 13 columns in laptop.csv. After cleaning: 1,231 laptops.

## Method (CRISP-ML(Q))
1. Data understanding
2. Preprocessing: datatypes, missing values, duplicates, outliers, feature engineering
   (PPI, SSD/HDD size, CPU family), one-hot encoding, scaling
3. Models: Linear Regression, SVR, KNN, Decision Tree, Random Forest (80/20 split)
4. Evaluation: MAE, MSE, RMSE, R-squared

## Results
Random Forest: R-squared about 0.79, mean absolute error about 10,400 rupees on unseen laptops.

## Files
- price-prediction.ipynb : full analysis with insights
- app.py : Streamlit web app
- laptop_model.joblib : trained model
- laptop.csv : dataset
