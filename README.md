# Data Modeling Analysis Description

This project focuses on analyzing and modeling the living wage in Ukraine from 2000 to 2023 using real data obtained by parsing the "Minfin.com.ua" website. The main goal is to create a mathematical model that accurately represents the changes in the living wage over time.

## Steps Involved

**Model Selection:** Choose between a linear trend with an exponential component and an anomalous error, or a quadratic trend with an exponential component and an anomalous error.

**Data Parsing:** Obtain real data from 'https://index.minfin.com.ua/ua/labour/wagemin/' and store it in 'Minfin_LivingWage.xlsx'.

**Statistical Characteristics:** Determine the statistical characteristics of the dataset.

**Anomaly Detection:** Detect and clean the dataset from anomalies.

**Model Quality Metrics:** Evaluate the quality of the models and optimize as needed.

**Statistical Learning (Linear Model):** Perform statistical learning of parameters using the polynomial model with the least squares method.

**Model Prediction (Linear):** Use the learned parameters to make predictions for the statistical series.

## Required Packages

The project utilizes the following Python packages:

pip version 23.1

numpy version 1.25.1 

pandas version 2.0.3

xlrd version 2.0.1

matplotlib version 3.7.2


## Model Selection

Two models are considered for the input data:

**Linear Model:** The statistical characteristics for this model include an exponential error distribution with parameters: n = 70, alfa = 60, a = 30, b = 5.

**Quadratic Model:** The statistical characteristics for this model also involve an exponential error distribution.

## **Data Characteristics**

For both the linear and quadratic models, the data characteristics are discussed, including the statistical distribution of the error and the confirmation of the trend with histograms.

## Anomaly Detection

The dataset is cleaned from anomalous measurements using the least squares method (MNK). Regressions are presented for both the linear and quadratic models.

## Model Quality Metrics

Various quality indicators for the models are provided, including the coefficient of determination (r-squared), global linear deviations, and the choice of the quadratic model based on these quality metrics.

## Statistical Learning

Statistical learning is performed for the cleaned dataset using the sliding window algorithm. A regression model is presented for this approach.

## Model Prediction

Using the learned parameters, predictions are made for the statistical series. Confidence intervals for forecasted values are also presented.

## Conclusion

The project concludes that the chosen model, the quadratic model, is adequate for representing the living wage in Ukraine. The entire process includes data cleaning, statistical learning, and parameter prediction, providing a comprehensive analysis of the dataset.

_Please note that this readme file provides a high-level overview of the project's steps and findings. Detailed code and data can be found in the project files._