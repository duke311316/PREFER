#!/usr/bin/env python3
# Description: This file is used to implement the flood prediction model
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import logging;
import os;
import time;
from datetime import date, datetime;
from flood_prediction.flood_pred_data_processing_V1 import DataProcessor;
from flood_prediction.flood_pred_models_V1 import FloodModel;

model_batch_size = 5 # number of time steps in the model
Processor = DataProcessor() 
Model = FloodModel(Processor)  
model_df = Processor. build_model_matrix() 
train_data, test_data = Model.split_df(model_df, model_batch_size
)
Model.inspect_data(model_df)
train_features, train_labels = Model.split_data_from_labels(train_data, model_batch_size
)
test_features, test_labels = Model.split_data_from_labels(test_data, model_batch_size
)
train_feature_normalizer = Model.model_data_normalizer(train_features)
tf_regressive_model = Model.tf_regressive_model(train_features)
tf_regressive_model.summary()
history = Model.history(tf_regressive_model, train_features, train_labels)





