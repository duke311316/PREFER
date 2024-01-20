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
    
    def create_lstm_dataset(self, data, n_future, n_past):
        Xs, ys = [], []
        for i in range(n_past, len(data) - n_future):
            Xs.append(data[i - n_past:i, :])
            ys.append(data[i + n_future, -1])  # Predicting the water level 4 hours ahead
        return np.array(Xs), np.array(ys)


    
    def create_lstm_dataset_include_water(self, data, n_future, n_past):
        Xs, ys = [], []
        for i in range(n_past, len(data) - n_future + 1):
            # Include past data up to the current point as features, including the current water level
            Xs.append(data[i - n_past:i, :])

            # Use the future water level as the target for prediction
            ys.append(data[i + n_future - 1, -1])  # Assuming the water level is the last column
        return np.array(Xs), np.array(ys)
    
    def build_lstm_model(self, input_shape):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=False))
        model.add(tf.keras.layers.Dense(units=25, activation='relu'))
        model.add(tf.keras.layers.Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
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
        output = Dense(1)(dense_layer)  # Predicting a single value

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model


    def train_model(self, model, X_train, y_train, epochs=10, batch_size=32, validation_split=0.1):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
        return history

    def predict_future(self, model, X_test):
        predictions = model.predict(X_test)
        return predictions


    def history(self, model, train_features, train_labels):
        history = model.fit(train_features, train_labels, epochs=20, batch_size=16, verbose=1, validation_split=0.2)
        return history

    
    def create_sklearn_dataset(self, data, n_future, n_past):
        Xs, ys = [], []
        for i in range(n_past, len(data) - n_future):
            # Create lagged features for past n_past steps
            lag_features = data[i - n_past:i, :-1].flatten()  # Flatten the lagged features
            Xs.append(lag_features)

            # Target variable
            ys.append(data[i + n_future, -1])  # Predicting the water level 4 hours ahead
        return np.array(Xs), np.array(ys)
    
    def build_linear_regression_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=[input_shape])
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    
    
    
    def sklearn_autoregressive_linear_model(self, X_train, y_train, X_test, y_test):
        # Create and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Making predictions
        predictions = model.predict(X_test)

        # Evaluating the model
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        print(f'Scikit-learn Autoregressive Linear Model - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}')

        return model, mae, mse, rmse, r2, predictions


    def plot_loss(self, history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [water_level]')
        plt.plot(hist['epoch'], hist['loss'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_loss'], label='Val Error')
        plt.legend()
        
    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        print(f'Mean Absolute Error: {mae}')
        print(f'Mean Squared Error: {mse}')
        print(f'Root Mean Squared Error: {rmse}')
        print(f'R² Score: {r2}')
        return mae, mse, rmse, r2, predictions
    
    def plot_model_predictions(self, y_test, predictions, model_names, target):
        """
        Plots the predictions from different models against actual values.

        Parameters:
        y_test (numpy.array): Actual target values.
        predictions (list of numpy.array): List containing prediction arrays from different models.
        model_names (list of str): Names of the models corresponding to the predictions.
        target (int): Number of time steps ahead for the prediction (e.g., 16 for 4 hours ahead if each step is 15 minutes).
        """
        num_models = len(predictions)
        fig, ax = plt.subplots(num_models, 1, figsize=(10, num_models * 4))

        for i in range(num_models):
            # Check if predictions are 1D or 2D and plot accordingly
            if predictions[i].ndim == 1:
                ax[i].plot(predictions[i], label=f'{model_names[i]} Predicted Water Level', color='red', linestyle='dashed')
            elif predictions[i].ndim == 2:
                ax[i].plot(predictions[i][:, 0], label=f'{model_names[i]} Predicted Water Level', color='red', linestyle='dashed')
            else:
                raise ValueError("Unsupported prediction array shape.")

            ax[i].plot(y_test, label='Actual Water Level', color='blue', linestyle='solid')
            ax[i].set_title(f'Prediction for {target/4} hours in the future with {model_names[i]}')
            ax[i].legend()

        plt.tight_layout()
        plt.savefig(f'./results/predictions/{target/4}_hours.png')
        plt.show()
