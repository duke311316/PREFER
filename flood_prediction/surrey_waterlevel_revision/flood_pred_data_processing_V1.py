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
    def __init__(self, most_recent_date =  '2023-04-24' , most_recent_time ='00:00'):
        # most_recent_date_str = most_recent_date.strftime('%Y-%m-%d') # format the most_recent date as a string
        self.most_recent_date = pd.to_datetime('%s %s' % (most_recent_date, most_recent_time)) # convert the most_recent date to a datetime object
        self.end_date = '2023-06-29' # subtract 6 months from the most_recent date to get the end date
        self.end_time = '00:00'
        # Define the dataframes for weather underground and surrey street stream gauge data
        self.wu_df = pd.DataFrame()
        self.wu_station_df = pd.DataFrame()
        self.ss_df = pd.DataFrame()
        self.model_df = pd.DataFrame()
        self.wu_station_path = './datasets/KLALAFAY_25_Stations/'
        self.ss_stream_gauge_path = './datasets/Surrey_Street_Stream_Gauge/'
        self.freq = '15T' # 15 minute frequency
        self.station_number = None


    def format_datetime(self,df):
        date_df = pd.DataFrame()
        date_df['date_time'] = pd.to_datetime(df['ObsTimeLocal'], format='%Y-%m-%d %H:%M')
        return date_df
    

    # Here we can easily switch between training and testing model parameters bycommenting and uncommenting the appropriate lines 

    def wu_parameter_selection(self):
        # self.wu_station_df['humidity'] = self.wu_station_df['HumidityAvg']
        # self.wu_station_df['temperature'] = self.wu_station_df['imperial.tempAvg']
        self.wu_station_df['wind_speed'] = self.wu_station_df['imperial.windSpeedAvg']
        self.wu_station_df['wind_direction'] = self.wu_station_df['windDirAvg']
        self.wu_station_df['prec_total'] = self.wu_station_df['imperial.precipTotal']
        # self.wu_station_df['heat_index'] = self.wu_station_df['imperial.heatindexLow']
        # self.wu_station_df['wind_chill'] = self.wu_station_df['imperial.windchillLow']
        self.wu_station_df['id'] = self.wu_station_df['stationID']
        # self.wu_station_df['prec_rate'] = self.wu_station_df['imperial.precipRate']
        # self.wu_station_df['pressure'] = self.wu_station_df['imperial.pressureMax']
        # self.wu_station_df['dew_point'] = self.wu_station_df['imperial.dewptAvg']


#         drop_unneeded_columns(self) is called after parameterSelection() dropping all of the columns not included as parameters in the model. This is done to reduce the size of the dataframe and to make it easier to read.
    def drop_unneeded_columns(self):
        self.wu_station_df.drop(columns=['id','tz','imperial.windspeedAvg','uvHigh', 'stationID','Unnamed: 0', 'solarRadiationHigh', 'imperial.precipRate', 'imperial.precipTotal', 'imperial.pressureMax', 'lat', 'lon','imperial.dewptAvg','imperial.tempAvg','imperial.windgustAvg', 'obsTimeUtc', 'winddirAvg','humidityAvg', 'qcStatus', 'humidityHigh', 'imperial.tempHigh', 'imperial.tempLow', 'imperial.windchillLow', 'epoch','imperial.windgustLow', 'imperial.pressureTrend', 'imperial.windgustHigh', 'imperial.windspeedLow', 'humidityLow','imperial.windchillAvg', 'imperial.windchillHigh', 'imperial.windspeedHigh', "imperial.heatindexHigh", 'imperial.heatindexLow', 'imperial.heatindexAvg', 'imperial.dewptHigh', 'imperial.dewptLow', 'imperial.pressureMin',  ], inplace=True)

# process_surrey(self) reads the surrey street stream gauge data and formats the datetime column to be in the same format as the weather underground data. It also calls time_window(self) to ensure that the dataframes are the same size.
        
    def process_surrey(self):
        
        self.ss_df = pd.read_csv(self.ss_stream_gauge_path + 'vermilion_river_water_level.txt', delimiter='\t')
        self.ss_df.drop(self.ss_df.index[0], inplace=True) # drop the first row.
        print(self.ss_df.head())

        self.ss_df = self.ss_df.rename(columns={'63172_00065': 'water_level'}) # rename the column
        self.ss_df = self.ss_df[['datetime', 'water_level']] # select only the datetime and water_level columns
        self.ss_df['datetime'] = pd.to_datetime(self.ss_df['datetime'], format='%Y-%m-%d %H:%M') # format the datetime column
        self.time_window(self.ss_df)# ensure that the dataframes are the same size.
        
        self.ss_df.set_index('datetime', inplace=True) # set the datetime column as the index.
        self.ss_df = self.ss_df.asfreq(self.freq) # resample the dataframe to 15 minute intervals and take the mean of the water level column for each interval to reduce the size of the dataframe.
        self.ss_df['water_level'] = self.ss_df['water_level'].infer_objects(copy=False) # interpolate the water level column to fill any missing values with the average of the previous and next values in the column.
        return self.ss_df['water_level']


    # process_klalafay_station(self, station_number) reads the weather underground data for a specific station and calls parameter_selection(self)to format the data.  
    def process_klalafay_station(self, station_number):
        try:
            path = self.wu_station_path + 'KLALAFAY' + station_number + '.csv'
            self.wu_station_df = pd.read_csv(path)
            self.wu_parameter_selection()
        except Exception as e:
            logging.error('Error in process_wu(): %s' % e)
            sys.exit(1)


    #  time_window(self, df) ensures that the dataframes are the same size by removing any rows that are outside of the time window defined in the constructor. It need not be called directly as it is called by concat_multiple_wu_stations(self).
    def time_window(self, df):
        df = df[(df['datetime'] >= self.most_recent_date) & (df['datetime'] <= self.end_date)] 
        

    #  concat_multiple_wu_stations(self) concatenates the weather underground data for all of the stations in the KLALAFAY_25_Stations folder. It calls process_klalafay_station(self, station_number) to read the data for each station and then calls drop_unneeded_columns(self) to remove the columns that are not included as parameters in the model. It also calls time_window(self, df) to ensure that the dataframes are the same size. In time to have more accurate ground truth data, we will need to find a way to combine the data from many more of the stations in Louisiana.
        
    def concat_multiple_wu_stations(self):
        
       
        if os.path.exists(self.wu_station_path): # check if the path exists
            stations = os.listdir(self.wu_station_path)  # get the list of stations
            print("The dataframe is being concatenated") 
            for i , (station) in enumerate(stations): # iterate through the stations
                if station.split('Y')[0]=="KLALAFAY":  
                    self.station_number = station.split('Y')[1].split('.')[0] # get the station number
                    self.process_klalafay_station(self.station_number) # read the weather underground data for the station 
                    self.drop_unneeded_columns()
                if i == 0: 

                    date_time = self.format_datetime(self.wu_station_df) # format the datetime column
                    self.wu_station_df = pd.concat([date_time, self.wu_station_df], axis=1)
                    print( self.wu_station_df.head())
                # self.wu_station_df.drop(columns=['ObsTimeLocal'], inplace=True) # drop the unformatted datetime column
                for column in self.wu_station_df.columns:
                    self.wu_station_df.rename(columns={column: column + '_' + self.station_number}) # rename the columns to include the station number
                    print(column)
                self.wu_df = pd.concat([self.wu_df, self.wu_station_df], axis=1) # concatenate the weather underground data for each station
                self.wu_df = self.wu_df.sort_values(by=['datetime']) # sort the dataframe by datetime column in ascending order.
                self.wu_df = self.time_window(self.wu_df) # ensure that the dataframes are the same size
                self.wu_df = self.wu_df.reset_index(drop=True) # reset the index
                self.wu_df = self.wu_df.fillna(method='ffill') # fill any missing values with the previous value
                self.wu_df = self.wu_df.resample(self.freq).mean() # resample the dataframe to 15 minute intervals and take the mean of the water level column for each interval to reduce the size of the dataframe.
                return self.wu_df
        else: print("not found")

    def build_model_matrix(self):
        window = 50 # window size for rolling mean
        self.model_df = pd.concat([self.wu_df, self.ss_df['water_level']], axis=1) # concatenate the weather underground and surrey street stream gauge data
        self.model_df = self.model_df.rolling(window=window).mean() # take the rolling mean of the dataframe to smooth the data
        self.model_df = self.model_df.dropna(how='any') # drop any rows with missing values
        self.model_df.to_csv('./datasets/model_df.csv', index=False) # save the dataframe to a csv file
        return self.model_df # return the dataframe

   
             


