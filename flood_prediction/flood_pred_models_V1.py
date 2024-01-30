import random
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Normalization, Input, Dense, LSTM, Bidirectional, Flatten, Activation, RepeatVector, Multiply, Permute, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from flood_pred_data_processing_V1 import DataProcessor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


class FloodModel:

    def __init__(self):
        
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.train_features = None

# split_data_from_labels () is a function that splits a DataFrame into training and testing sets while ensuring each set has enough data to create LSTM sequences. It takes in the DataFrame, the proportion of the dataset to include in the train split, and the number of past and future time steps to use for LSTM sequences. It returns the train and test sets as DataFrames.
    def split_data_from_labels(self, train_df, test_df):
        train_features = train_df.copy()
        train_labels = train_features.pop(['water_level', 'flow_rate'])  # Now handling two targets
        test_features = test_df.copy()
        test_labels = test_features.pop(['water_level', 'flow_rate'])  # Now handling two targets
        return train_features, train_labels, test_features, test_labels

#  split_df () is a function that splits a DataFrame into training and testing sets while ensuring each set has enough data to create LSTM sequences. It takes in the DataFrame, the proportion of the dataset to include in the train split, and the number of past and future time steps to use for LSTM sequences. It returns the train and test sets as DataFrames.
    def split_df(self, df, train_ratio=0.8, n_past=48, n_future=16):
        """
        Splits a DataFrame into training and testing sets while ensuring each set 
        has enough data to create LSTM sequences.

        :param df: DataFrame to split.
        :param train_ratio: Proportion of the dataset to include in the train split.
        :param n_past: Number of past time steps used for LSTM sequences.
        :param n_future: Number of future time steps to predict.
        :return: train_data, test_data as DataFrames.
        """
        total_length = len(df)
        train_length = int(total_length * train_ratio)

        # Ensure there's enough data for at least one sequence in the test set
        if total_length - train_length < n_past + n_future:
            train_length = total_length - (n_past + n_future)

        train_data = df.iloc[:train_length]
        test_data = df.iloc[train_length - n_past:]  # Overlap by n_past to allow sequence creation in test set

        return train_data, test_data

# model_data_normalizer () is a function that creates a normalization layer adapted to the given DataFrame. It takes in the DataFrame and returns the normalization layer.
    def model_data_normalizer(self, df):
        """
        Creates a normalization layer adapted to the given DataFrame.

        This function computes the mean and standard deviation of each feature
        in the DataFrame and returns a Normalization layer configured with these
        values, which can then be used to normalize the data.

        Parameters:
        df (pandas.DataFrame): The DataFrame whose features are to be normalized.

        Returns:
        tensorflow.keras.layers.Layer: A Normalization layer adapted to the DataFrame.
        """
        normalizer = Normalization(axis=-1)
        normalizer.adapt(df.to_numpy())
        return normalizer
    
    # normalize_data () is a function that normalizes the data using the data normalizer. It takes in the data and returns the normalized data.
    def normalize_data(self, df):
        normalizer = self.model_data_normalizer(df)
        normalized_data = normalizer(df)
        return normalized_data.numpy()
    

    def tf_regressive_model(self, normalizer):
        model = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(1)
        ])
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
        return model
    
    
    # tf_nn_model () is a function that creates a neural network model using TensorFlow. It takes in the data normalizer and returns the model. 
    def tf_nn_model(self, normalizer):
        model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(100, activation='sigmoid'),
        tf.keras.layers.Dropout(.3),  # Dropout can be adjusted
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(1)
    ])
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.005))
        return model
    
#  create_lstm_dataset () is a function that creates a dataset for LSTM models. It takes in the data, the number of time steps to predict in the future, and the number of time steps to look back. It returns the features and labels for the LSTM model.
    def create_lstm_dataset(self, data, n_future, n_past):
        Xs, ys = [], []
        for i in range(n_past, len(data) - n_future + 1):
            Xs.append(data[i - n_past:i, :-2])  # Excluding target variables
            ys.append(data[i + n_future - 1, -2:])  # Last two columns are targets
        return np.array(Xs), np.array(ys)

    #  create_lstm_dataset_include_water () is a function that creates a dataset for LSTM models. It takes in the data, the number of time steps to predict in the future, and the number of time steps to look back. It returns the features and labels for the LSTM model.
    def create_lstm_dataset_include_water(self, data, n_future, n_past):
        Xs, ys = [], []
        for i in range(n_past, len(data) - n_future + 1):
            # Include past data up to the current point as features, including the current water level
            Xs.append(data[i - n_past:i, :])

            # Use the future water level as the target for prediction
            ys.append(data[i + n_future - 1, -1])  # Assuming the water level is the last column
        return np.array(Xs), np.array(ys)
    

    # build_lstm_model () is a function that creates a Bidirectional LSTM model. It takes in the input shape and returns the model.
    def build_lstm_model(self, input_shape):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=False))
        model.add(tf.keras.layers.Dense(units=25, activation='relu'))
        model.add(tf.keras.layers.Dense(units=2))  # Output layer for two predictions
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

#  build_lstm_model_w_attention () is a function that creates a Bidirectional LSTM model with attention. It takes in the input shape and returns the model. 
    def build_lstm_model_w_attention(self, input_shape):
        # Define the input layer
        inputs = Input(shape=input_shape)

        # Add a Bidirectional LSTM layer
        lstm_out = Bidirectional(LSTM(40, return_sequences=True))(inputs)

        # Attention Mechanism - Start
        attention = Dense(1, activation='tanh')(lstm_out)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(40*2)(attention)
        attention = Permute([2, 1])(attention)
        sent_representation = Multiply()([lstm_out, attention])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(40*2,))(sent_representation)
        # Attention Mechanism - End

        # Final Dense layers
        dense_layer = Dense(25, activation='relu')(sent_representation)
        output = Dense(2)(dense_layer)  # Adjusted to predict two values

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)
        return model


# train_model () is a function that trains the given model on the given data. It takes in the model, the training data, the number of epochs, the batch size, and the validation split. It returns the history of the training process.
    def train_model(self, model, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
        return history

# predict_future () is a function that makes predictions on the given data using the given model. It takes in the model and the test data and returns the predictions.
    def predict_future(self, model, X_test):
        predictions = model.predict(X_test)
        return predictions
    
    #  create_sklearn_dataset () is a function that creates a dataset for scikit-learn models. It takes in the data, the number of time steps to predict in the future, and the number of time steps to look back. It returns the features and labels for the scikit-learn model.
    def create_sklearn_dataset(self, data, n_future, n_past):
        Xs, ys = [], []
        for i in range(n_past, len(data) - n_future):
            # Create lagged features for past n_past steps
            lag_features = data[i - n_past:i, :-1].flatten()  # Flatten the lagged features
            Xs.append(lag_features)

            # Target variable
            ys.append(data[i + n_future, -1])  # Predicting the water level 4 hours ahead
        return np.array(Xs), np.array(ys)
    
    def create_sklearn_dataset(self, data, n_future, n_past):
        Xs, ys = [], []
        for i in range(n_past, len(data) - n_future):
            # Create lagged features for past n_past steps
            lag_features = data[i - n_past:i, :-2].flatten()  # Excluding the last two columns (targets)
            Xs.append(lag_features)

            # Target variables
            ys.append(data[i + n_future, -2:])  # Predicting both water level and flow rate 4 hours ahead
        return np.array(Xs), np.array(ys)

    # build_linear_regression_model () is a function that creates a linear regression model using TensorFlow. It takes in the input shape and returns the model.
    def build_linear_regression_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=[input_shape])
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

#  sklearn_autoregressive_linear_model () is a function that creates a linear regression model using scikit-learn and trains it on the given data. It returns the trained model and the evaluation metrics. 
    # def sklearn_autoregressive_linear_model(self, X_train, y_train, X_test, y_test):
    #     # Create and train the model
    #     model = LinearRegression()
    #     model.fit(X_train, y_train)

    #     # Making predictions
    #     predictions = model.predict(X_test)

    #     # Evaluating the model
    #     mae = mean_absolute_error(y_test, predictions)
    #     mse = mean_squared_error(y_test, predictions)
    #     rmse = np.sqrt(mse)
    #     r2 = r2_score(y_test, predictions)

    #     print(f'Scikit-learn Autoregressive Linear Model - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}')

    #     return model, mae, mse, rmse, r2, predictions


# sklearn_autoregressive_linear_model () is a function that creates a linear regression model using scikit-learn and trains it on the given data. It returns the trained model and the evaluation metrics.
    def sklearn_autoregressive_linear_model(self, X_train, y_train, X_test, y_test):
        # Create and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Making predictions
        predictions = model.predict(X_test)

        # Evaluating the model for each output
        for i, target in enumerate(['Water Level', 'Flow Rate']):
            mae = mean_absolute_error(y_test[:, i], predictions[:, i])
            mse = mean_squared_error(y_test[:, i], predictions[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test[:, i], predictions[:, i])
            print(f'Target: {target} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}')

        return model, predictions
    
# plot_loss () is a function that plots the loss of the model during training. It takes in the history of the training process and plots the loss.
    def plot_loss(self, history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [water_level]')
        plt.plot(hist['epoch'], hist['loss'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_loss'], label='Val Error')
        plt.legend()
        
    # def evaluate_model(self, model, X_test, y_test):
    #     predictions = model.predict(X_test)
    #     mae = mean_absolute_error(y_test, predictions)
    #     mse = mean_squared_error(y_test, predictions)
    #     rmse = np.sqrt(mse)
    #     r2 = r2_score(y_test, predictions)
    #     print(f'Mean Absolute Error: {mae}')
    #     print(f'Mean Squared Error: {mse}')
    #     print(f'Root Mean Squared Error: {rmse}')
    #     print(f'R² Score: {r2}')
    #     return mae, mse, rmse, r2, predictions
    
    # evaluate_model () is a function that evaluates the model on the test data. It takes in the model, the test data, and the test labels. It returns the evaluation metrics and the predictions.
    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        for i, target_name in enumerate(['Water Level', 'Flow Rate']):
            mae = mean_absolute_error(y_test[:, i], predictions[:, i])
            mse = mean_squared_error(y_test[:, i], predictions[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test[:, i], predictions[:, i])
            print(f'Target: {target_name} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}')

        return predictions  # Returns the predictions for further analysis if needed
    
    # plot_model_predictions () is a function that plots the predictions from different models against actual values for the entire dataset. It takes in the actual target values, the predictions, and the names of the models corresponding to the predictions.
    def plot_model_predictions(self, y_test, predictions, model_names):
        """
        Plots the predictions from different models against actual values for the entire dataset.

        Parameters:
        y_test (numpy.array): Actual target values.
        predictions (list of numpy.array): List containing prediction arrays from different models.
        model_names (list of str): Names of the models corresponding to the predictions.
        """
        num_models = len(predictions)
        fig, axs = plt.subplots(num_models, 2, figsize=(15, num_models * 4))  # 2 columns for Water Level and Flow Rate

        for i in range(num_models):
            for j in range(2):  # Loop through Water Level (0) and Flow Rate (1)
                axs[i, j].plot(y_test[:, j], label='Actual', color='black', linestyle='solid')
                axs[i, j].plot(predictions[i][:, j], label=f'{model_names[i]} Predicted', color='red', linestyle='solid')

                axs[i, j].set_title(f'{model_names[i]} - {"Water Level" if j == 0 else "Flow Rate"}')
                axs[i, j].set_xlabel('Time Steps')
                axs[i, j].set_ylabel('River Water Level' if j == 0 else 'Flow Rate')
                axs[i, j].legend()

        plt.tight_layout()
        plt.show()
        
        # plot_4_hour_prediction () is a function that plots the predictions from different models against actual values for a 4-hour period. It takes in the actual target values, the predictions, the names of the models corresponding to the predictions, the target time steps, and the time interval between predictions in minutes.
    def plot_4_hour_prediction(self, y_test, predictions, model_names,target = 16, interval_minutes=15):
        """
        Plots the predictions from different models against actual values for a 4-hour period.

        Parameters:
        y_test (numpy.array): Actual target values.
        predictions (list of numpy.array): List containing prediction arrays from different models.
        model_names (list of str): Names of the models corresponding to the predictions.
        interval_minutes (int): Time interval between predictions in minutes.
        """
        num_models = len(predictions)
        fig, axs = plt.subplots(num_models, 2, figsize=(15, num_models * 4))  # 2 columns for Water Level and Flow Rate
        
        # Time labels and ticks for a 4-hour period at 30-minute intervals
        time_labels = [f"{(interval_minutes * i) // 60:02d}:{(interval_minutes * i) % 60:02d}" for i in range(0, 16, 2)]
        time_ticks = np.arange(0, 16, 2)  # Positions for the time labels every 30 mins

        for i in range(num_models):
            for j in range(2):  # Loop through Water Level (0) and Flow Rate (1)
                axs[i, j].plot(y_test[:16, j], label='Actual', color='black', linestyle='solid')
                axs[i, j].plot(predictions[i][:16, j], label=f'{model_names[i]} Predicted', color='red', linestyle='solid')

                axs[i, j].set_title(f'{model_names[i]} - {"Water Level" if j == 0 else "Flow Rate"}')
                axs[i, j].set_xlabel('Time Steps')
                axs[i, j].set_ylabel('River Water Level' if j == 0 else 'Flow Rate')
                axs[i, j].set_xticks(time_ticks)  # Set positions for the time labels
                axs[i, j].set_xticklabels(time_labels)  # Set time labels
                axs[i, j].legend()

        plt.tight_layout()
        plt.savefig(f'./results/predictions/{target/4}_hours_1_period.png')
        plt.show()  
        