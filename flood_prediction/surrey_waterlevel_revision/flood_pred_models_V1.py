import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Normalization
from flood_pred_data_processing_V1 import DataProcessor


class FloodModel:

    def __init__(self):
        
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.train_features = None

    def split_df(self, df, target):
        total_length = len(df)
        train_length = int(total_length * 0.8)
        train_length -= train_length % target
        test_length = train_length - ((len(df) - train_length) % target)
        train_data = df.iloc[:train_length]
        test_data = df.iloc[test_length:]
        return train_data, test_data

    def inspect_data(self, df):
        print(df.head())
        print(df.tail())
        print(df.shape)
        print(df.describe().transpose())
        print(df.isnull().sum())
        print(df.info())
        print(df.dtypes)
        sns.pairplot(df)
        plt.show()

    def split_data_from_labels(self, train_df,test_df):
        train_features = train_df.copy()
        train_labels = train_features.pop('water_level')
        test_features = test_df.copy()
        test_labels = test_features.pop('water_level')
        return train_features, train_labels, test_features, test_labels

    def model_data_normalizer(self, df):
        normalizer = Normalization(axis=-1)
        normalizer.adapt(np.array(df))
        return normalizer

    def tf_regressive_model(self, normalizer):
        model = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(1)
        ])
        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.01))
        return model
    
    def tf_nn_model(self, normalizer):
        model = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(10),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Dense(1)
        ])
        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def history(self, model, train_features, train_labels):
        history = model.fit(train_features, train_labels, epochs=100, batch_size=5, verbose=1, validation_split=0.2)
        return history

    def plot_loss(self, history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [water_level]')
        plt.plot(hist['epoch'], hist['loss'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_loss'], label='Val Error')
        plt.legend()