# PREFER

Flood Prediction Using Machine Learning

by: Dale Duclion

Mentors: Bryce Turney,

Overview
The data processing was successful though the use of the data was not.
This project focuses on predicting water levels using machine learning techniques. The primary goal is to develop models that can accurately forecast future water levels based on historical data. The research journey encompasses transitioning from linear autoregressive models to more sophisticated deep learning approaches, such as LSTM (Long Short-Term Memory) networks, with the inclusion of an attention mechanism for improved accuracy.

Datasets

The primary dataset used in this project is the Surrey St. Stream Gauge data, sourced from USGS Water Data, and data from select Weather Underground weather station in the Lafayette Area. This dataset provides critical information on water levels, which is essential for our predictive models.

DataProcessor Class

DataProcessor is responsible for reading, processing, and preparing the dataset for modeling. 

Key functionalities include:

Reading and formatting data from Surrey Street Stream Gauge and Weather Underground datasets.
Handling date-time conversions to maintain consistency.
Selecting relevant parameters from the dataset for model input.
Concatenating data from multiple weather stations to enrich the dataset.
Implementing methods for data normalization and windowing.

FloodModel Class

FloodModel encapsulates various machine learning and deep learning models for flood prediction. Key features include:

Splitting data into training and testing sets.
Data normalization using TensorFlow's Normalization layer.
Implementing a linear regression model as a baseline.
Developing LSTM models for time-series forecasting.
Enhancing LSTM with an attention mechanism for improved performance.
Training and evaluating models using metrics like MAE, MSE, RMSE, and R².
Visualization of model predictions and performance.

Main Script Execution

The main script (run.ipynb) performs the following steps:

Initializes the DataProcessor to process the Surrey and Weather Underground data.
Normalizes features and prepares datasets for LSTM and linear regression models.
Trains and evaluates LSTM and linear regression models on the prepared dataset.
Compares model predictions and visualizes the results.

Requirements

To run this project, the following libraries and frameworks are required:

Python 3.x
Pandas
NumPy
Matplotlib
Seaborn
TensorFlow
Scikit-Learn

Usage

Clone the repository to your local machine.
Ensure that all required libraries are installed.
Run the run.ipynb notebook to execute the data processing, model training, and evaluation.

Acknowledgements

This project could not have been made possible if not for Bryce Turney's mentorship and the assistance of ChatGPT for troubleshooting and refining the machine learning models, particularly in areas such as code optimization and implementation of advanced neural network architectures.


