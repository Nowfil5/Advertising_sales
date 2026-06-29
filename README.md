# Sales Prediction via TV Advertising (OLS Regression)

## Project Overview
This repository contains a machine learning project that applies Ordinary Least Squares (OLS) regression to quantify the impact of TV advertising budgets on product sales. The goal is to provide actionable, data-driven insights for marketing budget optimization.

## Data Structure & Methodology
The project utilizes a dataset of 200 observations. The model establishes a simple linear regression mapping the independent variable (TV ad spend) to the dependent variable (Sales). 

**Core Technologies:**
* Python 3
* `statsmodels` for OLS implementation
* `pandas` for data manipulation
* `matplotlib` / `seaborn` for visualization

## Results & Statistical Significance
The model demonstrates a strong predictive capability with the following key metrics:
* **R-squared:** 0.812 (Explains 81.2% of the variance in sales).
* **F-statistic:** 856.2 (Statistically significant at p < 0.001).
* **Base Sales (Intercept):** 6.97 units when TV ad spend is zero.
* **TV Coefficient:** An increase of 0.0555 units in sales for every single unit increase in TV ad spend.

## System Requirements & Setup
This project is developed and optimized for an Ubuntu Linux environment. 

1. Clone the repository:
   git clone https://github.com/Nowfil5/Advertising_sales.git
2. Navigate to the directory:
   cd Advertising_sales
3. Install dependencies:
   pip install -r requirements.txt
4. Run the primary analysis script:
   python main.py
