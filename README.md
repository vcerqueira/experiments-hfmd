# Hand-Foot-Mouth Disease Forecasting Benchmark

This repository contains code for benchmarking different forecasting models for predicting hand-foot-mouth disease (HFMD) cases. We evaluate various approaches including classical time series methods, machine learning models including deep neural networks and gradient boosting, and specialized count data techniques.

## Research Questions

We address three key research questions:

1. **Model Performance Comparison**: What is the most accurate forecasting method among:
   - Classical forecasting techniques (ARIMA, Croston)
   - Machine learning and deep learning methods
   - Zero-inflated regression models trained based on auto-regression

2. **Impact of Meteorological Variables**: How do exogenous meteorological variables affect model performance?

3. **Global vs Local Training**: Is it better to train models on data from multiple regions (global approach) or only on the target region's historical data (local approach)?

## Dataset

The dataset used in this study contains HFMD case counts and meteorological variables across different regions. Due to privacy considerations, the raw data is not included in this repository.

## Code Structure

The experiments are based on the Nixtla framework.

To install the required dependencies, run `pip install -r requirements.txt` 


- All experiment execution scripts are located in `scripts/experiments/run/`
- Results analysis and visualization scripts can be found in `scripts/experiments/analysis/`



## Results

The following figure shows the SMAPE across different forecasting models:

![Model Performance Comparison](./assets/outputs/results_main.png)

AutoNHITS, using a Poisson based objective functions, performs best overall

