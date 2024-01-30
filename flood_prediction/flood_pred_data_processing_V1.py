# surrey st stream gauge data source: https://waterdata.usgs.gov/monitoring-location/07386880/#parameterCode=72254&period=P365D&showMedian=false



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import time
from datetime import date, datetime, timedelta
import sys


# sci-kit learn dependencies
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error


# Tensorflow dependencies
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns


from dateutil.relativedelta import relativedelta # used to subtract months to the most_recent date 

# The DataProcessor class is used to read and format the data from the weather underground and surrey street stream gauge datasets. It also contains methods for selecting the parameters to be used in the model and for concatenating the weather underground data for all of the stations in the KLALAFAY_25_Stations folder.
class DataProcessor:
    def __init__(self, most_recent_date =  '2023-04-01' , most_recent_time ='00:00'):
        # most_recent_date_str = most_recent_date.strftime('%Y-%m-%d') # format the most_recent date as a string
        self.most_recent_date = pd.to_datetime('%s %s' % (most_recent_date, most_recent_time)) # convert the most_recent date to a datetime object
        self.end_date = '2023-06-29' # subtract 6 months from the most_recent date to get the end date
        self.end_time = '00:00'
        # Define the dataframes for weather underground and surrey street stream gauge data
        self.wu_df = pd.DataFrame()
        self.wu_station_df = pd.DataFrame()
        self.ss_wl_df = pd.DataFrame()
        self.ss_fr_df = pd.DataFrame()
        self.model_df_wo_surrey = pd.DataFrame()
        self.model_df_w_surrey = pd.DataFrame()

        self.wu_station_path = "./datasets/KLALAFAY_25_Stations/"
        self.ss_stream_gauge_path = './datasets/Surrey_Street_Stream_Gauge/'
        self.freq = '15T' # 15 minute frequency
        self.station_number = None


    def format_datetime(self, df):
        if 'obsTimeLocal' not in df.columns:
            print("Column 'obsTimeLocal' not found in DataFrame.")
            return pd.DataFrame()

        try:
            date_df = pd.DataFrame()
            date_df['datetime'] = pd.to_datetime(df['obsTimeLocal'], format='%m/%d/%y %H:%M')
            return date_df
        except Exception as e:
            print(f"Error formatting datetime: {e}")
            return pd.DataFrame()
    

    # Here we can easily switch between training and testing model parameters by commenting and uncommenting the appropriate lines 

    def wu_parameter_selection(self):
        self.wu_station_df['humidity'] = self.wu_station_df['humidityAvg']
        self.wu_station_df['temperature'] = self.wu_station_df['imperial.tempAvg']
        self.wu_station_df['wind_speed'] = self.wu_station_df['imperial.windspeedAvg']
        self.wu_station_df['wind_direction'] = self.wu_station_df['winddirAvg']
#     ###  Due to poor measurements, the prec_total should never be used but is 
#          left for versioning purposes ###
#         self.wu_station_df['prec_total'] = self.wu_station_df['imperial.precipTotal']
        # self.wu_station_df['heat_index'] = self.wu_station_df['imperial.heatindexLow']
        # self.wu_station_df['wind_chill'] = self.wu_station_df['imperial.windchillLow']
        self.wu_station_df['id'] = self.wu_station_df['stationID']
        self.wu_station_df['prec_rate'] = self.wu_station_df['imperial.precipRate']
        # self.wu_station_df['pressure'] = self.wu_station_df['imperial.pressureMax']
        self.wu_station_df['dew_point'] = self.wu_station_df['imperial.dewptAvg']


#         drop_unneeded_columns(self) is called after parameterSelection() dropping all of the columns not included as parameters in the model. This is done to reduce the size of the dataframe and to make it easier to read.
    def drop_unneeded_columns(self):
        self.wu_station_df.drop(columns=['obsTimeLocal','id','tz','imperial.windspeedAvg','uvHigh', 'stationID','Unnamed: 0', 'solarRadiationHigh', 'imperial.precipRate', 'imperial.precipTotal', 'imperial.pressureMax', 'lat', 'lon','imperial.dewptAvg','imperial.tempAvg','imperial.windgustAvg', 'obsTimeUtc', 'winddirAvg','humidityAvg', 'qcStatus', 'humidityHigh', 'imperial.tempHigh', 'imperial.tempLow', 'imperial.windchillLow', 'epoch','imperial.windgustLow', 'imperial.pressureTrend', 'imperial.windgustHigh', 'imperial.windspeedLow', 'humidityLow','imperial.windchillAvg', 'imperial.windchillHigh', 'imperial.windspeedHigh', "imperial.heatindexHigh", 'imperial.heatindexLow', 'imperial.heatindexAvg', 'imperial.dewptHigh', 'imperial.dewptLow', 'imperial.pressureMin',  ], inplace=True)
 #  time_window(self, df) ensures that the dataframes are the same size by removing any rows that are outside of the time window defined in the constructor. It need not be called directly as it is called by concat_multiple_wu_stations(self).
    def time_window(self, df):
        df = df[(df['datetime'] <= self.most_recent_date) & (df['datetime'] >= self.end_date)] 
        

# rename_columns_with_station_number(self, df, station_number) is called after drop_unneeded_columns(self) to rename the columns to include the station number. This is done to make it easier to read.
    def rename_columns_with_station_number(self, df, station_number):
        new_column_names = {col: f"{station_number}_{col}" for col in df.columns}
        return df.rename(columns=new_column_names)

# process_surrey(self) reads the surrey street stream gauge data and formats the datetime column to be in the same format as the weather underground data. It also calls time_window(self) to ensure that the dataframes are the same size.
        
    def process_surrey_water_level(self):
        self.ss_wl_df = pd.read_csv(self.ss_stream_gauge_path + 'vermilion_river_water_level.txt', delimiter='\t')
        self.ss_wl_df.drop(self.ss_wl_df.index[0], inplace=True) # drop the first row.
        self.ss_wl_df['datetime'] = pd.to_datetime(self.ss_wl_df['datetime'], format='%Y-%m-%d %H:%M')
        self.ss_wl_df.rename(columns={'63172_00065': 'water_level'}, inplace=True) 
        self.ss_wl_df = self.ss_wl_df[['datetime', 'water_level']] 
        self.time_window(self.ss_wl_df)# ensure that the dataframes are the same size.
        self.ss_wl_df.set_index('datetime', inplace=True)
        self.ss_wl_df = self.ss_wl_df.asfreq(self.freq)
        return self.ss_wl_df
    
    # process_surrey_flow_rate(self) reads the surrey street stream gauge data and formats the datetime column to be in the same format as the weather underground data. It also calls time_window(self) to ensure that the dataframes are the same size.
    def process_surrey_flow_rate(self):
            self.ss_fr_df = pd.read_csv(self.ss_stream_gauge_path + 'flow.txt', delimiter='\t')
            self.ss_fr_df.drop(self.ss_fr_df.index[0], inplace=True) 
            self.ss_fr_df['datetime'] = pd.to_datetime(self.ss_fr_df['datetime'], format='%Y-%m-%d %H:%M')
            self.ss_fr_df.rename(columns={'232928_72254': 'flow_rate'}, inplace=True)
            self.ss_fr_df = self.ss_fr_df[['datetime', 'flow_rate']] 
            self.time_window(self.ss_fr_df)
            self.ss_fr_df.set_index('datetime', inplace=True)
            self.ss_fr_df = self.ss_fr_df.asfreq(self.freq)
            return self.ss_fr_df

    
    # process_klalafay_station(self, station_number) reads the weather underground data for a specific station and calls parameter_selection(self)to format the data.  
    def process_klalafay_station(self, station_number):
        try:
            path = self.wu_station_path + 'KLALAFAY' + station_number + '.csv'
            self.wu_station_df = pd.read_csv(path)
            self.wu_parameter_selection()
        except Exception as e:
            logging.error('Error in process_wu(): %s' % e)
            sys.exit(1)


   
    #  concat_multiple_wu_stations(self) concatenates the weather underground data for all of the stations in the KLALAFAY_25_Stations folder. It calls process_klalafay_station(self, station_number) to read the data for each station and then calls drop_unneeded_columns(self) to remove the columns that are not included as parameters in the model. It also calls time_window(self, df) to ensure that the dataframes are the same size. In time to have more accurate ground truth data, we will need to find a way to combine the data from many more of the stations in Louisiana.
        
    def concat_multiple_wu_stations(self):
        if os.path.exists(self.wu_station_path): # check if the path exists
            stations = os.listdir(self.wu_station_path)  # get the list of stations
            print("The dataframe is being concatenated") 
            for i , (station) in enumerate(stations): # iterate through the stations
                if station.split('Y')[0]=="KLALAFA":  
                    self.station_number = station.split('Y')[1].split('.')[0] # get the station number
                    self.process_klalafay_station(str(self.station_number)) # read the weather underground data for the station 
                    
                    if i == 0: 
                        ("I = 0")

                        date_time = self.format_datetime(self.wu_station_df) # format the datetime column
                       
                        self.wu_df = date_time # set the datetime column as the index.
                    self.drop_unneeded_columns()    

                    self.wu_station_df = self.rename_columns_with_station_number(self.wu_station_df, self.station_number)
                else: continue
            
                self.wu_df = pd.concat([self.wu_df, self.wu_station_df], axis=1) # concatenate the weather underground data for each station
                
            self.wu_df = self.wu_df.sort_values(by=['datetime']) # sort the dataframe by datetime column in ascending order.
            self.time_window(self.wu_df) # ensure that the dataframes are the same size
            self.wu_df.set_index('datetime', inplace=True) # set the datetime column as the index.
            self.wu_df = self.wu_df.ffill() # fill any missing values with the previous value
            self.wu_df = self.wu_df.resample(self.freq).mean() # resample the dataframe to 15 minute intervals and take the mean of the water level column for each interval to reduce the size of the dataframe.
            return self.wu_df
        else: print("not found")

# build_model_matrix_wo_predictions_as_parameters(self) concatenates the weather underground and surrey street stream gauge data and takes the rolling mean of the dataframe to smooth the data. It also drops any rows with missing values and saves the dataframe to a csv file.
    def build_model_matrix_wo_predictions_as_parameters(self):
        window = 48
        # Align the indexes
        self.model_df_wo_surrey = pd.concat([self.wu_df, self.ss_wl_df['water_level']], axis=1)
        self.model_df_wo_surrey = pd.concat([self.model_df_wo_surrey, self.ss_fr_df['flow_rate']], axis=1)
        self.model_df_wo_surrey = self.model_df_wo_surrey.rolling(window=window).mean()
        self.model_df_wo_surrey = self.model_df_wo_surrey.dropna(how='any')
        self.model_df_wo_surrey.to_csv('./datasets/model_df_wo_surrey.csv', index=False)
        return self.model_df_wo_surrey
    
    # inspect_data(self, df) prints the first 5 rows, last 5 rows, shape, summary statistics, number of missing values, and data types of the dataframe. It also plots a pairplot of the dataframe.
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

    # split_df(self, df, train_ratio=0.8) splits the dataframe into training and testing data. The default ratio is 80% training and 20% testing.
    def split_df(self, df, train_ratio=0.8):
        total_length = len(df)
        train_length = int(total_length * train_ratio)

        train_data = df.iloc[:train_length]
        test_data = df.iloc[train_length:]

        return train_data, test_data 
   
    #  build_model_matrix_w_predictions_as_parameters(self) is the same as build_model_matrix_wo_predictions_as_parameters(self) except that it includes the water level and flow rate as parameters in the model. This is done to see if the model can learn to predict the water level and flow rate from the weather underground data.
    def build_model_matrix_w_predictions_as_parameters(self): 
        window = 48
        # Align the indexes
        water_level = pd.DataFrame(self.ss_wl_df['water_level'])
        flow_rate = pd.DataFrame(self.ss_fr_df['flow_rate'])
        self.model_df_w_surrey = pd.concat([self.wu_df, water_level['water_level']], axis=1)
        self.model_df_w_surrey = pd.concat([self.model_df_w_surrey, flow_rate['flow_rate']], axis=1)
        # self.model_df_w_surrey.rename(columns={'water_level': 'water_level_param'}, inplace=True)
        # self.model_df_w_surrey.rename(columns={'flow_rate': 'flow_rate_param'}, inplace=True)
        self.model_df_w_surrey.rename(columns={'water_level': 'water_level_param', 'flow_rate': 'flow_rate_param'}, inplace=True)
        self.model_df_w_surrey = pd.concat([self.model_df_w_surrey, water_level['water_level']], axis=1)
        self.model_df_w_surrey = pd.concat([self.model_df_w_surrey, flow_rate['flow_rate']], axis=1)
        self.model_df_w_surrey = self.model_df_w_surrey.rolling(window=window).mean()
        self.model_df_w_surrey = self.model_df_w_surrey.dropna(how='any')
        self.model_df_w_surrey.to_csv('./datasets/model_df_w_surrey.csv', index=False)
        return self.model_df_w_surrey
    
    
    
   






   
             


