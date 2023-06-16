import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from conf import raw_temperature_data_path, raw_demand_data_path, merged_data_demand_path, cleaned_data_path


def merge_csv_files_in_folder(folder_path):
    # Get a list of all csv files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Initialize an empty list to hold DataFrames
    dfs = []

    # Loop over all csv files and read each one into a DataFrame
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, csv_file))
        dfs.append(df)

    # Concatenate all DataFrames in the list
    merged_df = pd.concat(dfs, ignore_index=True)

    return merged_df


def merge_demand_and_temp_data(demand_df, temp_df):

    # Convert SETTLEMENT_DATE to datetime format
    demand_df['SETTLEMENT_DATE'] = pd.to_datetime(demand_df['SETTLEMENT_DATE'])

    # Since SETTLEMENT_DATE doesn't have time, we assume it refers to the beginning of the day
    # and strip time from observation_dtg_utc
    temp_df['observation_dtg_utc'] = pd.to_datetime(temp_df['observation_dtg_utc']).dt.normalize()

    # Merge the datasets on the date columns
    merged_df = pd.merge(demand_df, temp_df, left_on='SETTLEMENT_DATE', right_on='observation_dtg_utc')
    return merged_df


if __name__ == '__main__':
    # Merge demand data into one csv file
    # merged_df = merge_csv_files_in_folder(os.path.join(os.curdir, raw_demand_data_path))
    file_path = os.path.join(os.curdir, merged_data_demand_path)
    file_name_full_path = os.path.join(file_path, 'demand_data.csv')
    # merged_df.to_csv(file_name_full_path)

    # Load the data
    demand_df = pd.read_csv(file_name_full_path)
    file_path
    tempd_df = pd.read_csv(os.path.join(os.curdir, raw_temperature_data_path))

    merged_df = merge_demand_and_temp_data(demand_df, tempd_df)

    # Convert 'SETTLEMENT_DATE' to datetime format (if it is not already)
    merged_df['SETTLEMENT_DATE'] = pd.to_datetime(merged_df['SETTLEMENT_DATE'])

    # Set 'SETTLEMENT_DATE' as the DataFrame index
    merged_df.set_index('SETTLEMENT_DATE', inplace=True)

    # De-seasonalize TSD
    result = seasonal_decompose(merged_df['TSD'], model='additive', period=48)  # added freq=48
    merged_df['TSD_de-seasonalized'] = result.trend

    # Drop any rows with missing values
    merged_df.dropna(inplace=True)

    # Create a linear regression model
    X = merged_df['temp_c'].values.reshape(-1, 1)  # Feature: temperature
    y = merged_df['TSD_de-seasonalized']  # Target: de-seasonalized TSD

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    plt.figure(figsize=(12, 6))
    plt.scatter(X, y, color='blue', label='Actual')  # Plot the actual data
    plt.plot(X, y_pred, color='red', label='Fitted line')  # Plot the fitted line
    plt.xlabel('Temperature')
    plt.ylabel('TSD_de-seasonalized')
    plt.title('Temperature vs. TSD (de-seasonalized)')
    plt.legend()
    plt.show()

    # y_pred = model.predict(X)

    # Now df contains the merged data

    # # Deseasonlize the data
    # decomposed = seasonal_decompose(df["TSD"])
    # trend = decomposed.trend
    # seasonal = decomposed.seasonal
    # residual = decomposed.resid
    #
    # # Clean and transform the data
    # df["TSD"] = trend + seasonal
    # df["temp_c"] = pd.to_numeric(df["temp_c"])
    #
    # # Train a machine learning model to predict the TSD column data
    # model = LinearRegression()
    # model.fit(df[["temp_c"]], df["TSD"])
    #
    # # Predict the TSD column data for the next day
    # next_day_temp = 6.7
    # next_day_tsd = model.predict([[next_day_temp]])[0]
    #
    # print("The predicted TSD for the next day is", next_day_tsd)