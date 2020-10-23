import re
import pandas as pd
from string import ascii_letters
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#https://seaborn.pydata.org/examples/many_pairwise_correlations.html

def define_heat_map(df):
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

def define_compressed_features_set_one(df, close_label = 'Adj Close', window=50):
    df['Return'] = df[close_label] / df[close_label].shift(1) -1
    df['Momentum'] = df[close_label] / df[close_label].shift(5) -1
    df['SMA'] = df[close_label] / df[close_label].rolling(window = window).mean() -1
    df['Bollinger'] = (df[close_label] - df['SMA'])/ (2.0*df[close_label].rolling(window = window).std())
    return df

def define_features_set_one(df, close_label = 'Adj Close', window=50):
    df['Return'] = df[close_label] - df[close_label].shift(1)
    df['Momentum'] = df[close_label] - df[close_label].shift(5)
    df['SMA'] = df[close_label].rolling(window = window).mean()
    df['Bollinger_high'] = df['SMA'] + 2.0*df[close_label].rolling(window = window).std()
    df['Bollinger_low'] = df['SMA'] - 2.0*df[close_label].rolling(window = window).std()
    return df

#https://financetrain.com/feature-selection-in-machine-learning/
def define_features_set_two(dataset, close_label = 'Adj Close', volume_label = 'Volume', window=50, rsi_window_length = 21):
    features = pd.DataFrame(index=dataset.index)
    features[close_label] = dataset[close_label]
    features[volume_label] = dataset[volume_label]
    features['ma7'] = dataset[close_label].rolling(window=7).mean()
    features['ma21'] = dataset[close_label].rolling(window=21).mean()
    
    # Create MACD
    features['26ema'] = dataset[close_label].ewm(span=26).mean()
    features['12ema'] = dataset[close_label].ewm(span=12).mean()
    features['MACD'] = (features['12ema']-features['26ema'])
 
    # Create Bollinger Bands
    features['20sd'] = dataset[close_label].rolling(20).std()
    features['upper_band'] = features['ma21'] + (features['20sd']*2)
    features['lower_band'] = features['ma21'] - (features['20sd']*2)
    
    # Create Exponential moving average
    features['ema'] = dataset[close_label].ewm(span=20).mean()
    
    # ROC Rate of Change
    N = dataset[close_label].diff(10)
    D = dataset[close_label].shift(10)
    features['ROC'] = N/D
    
    #RSI
    delta = features[close_label].diff()
    delta.bfill(inplace=True)
    print(delta.head())
    print(delta.describe())

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    roll_up1 = up.ewm(span=rsi_window_length).mean()
    roll_down1 = down.abs().ewm(span=rsi_window_length).mean()

    RS1 = roll_up1 / roll_down1
    features['RSI'] = 100.0 - (100.0 / (1.0 + RS1))

    return features

def split_data(df, prediction_length, close_label = 'Adj Close'):
    df_X = df.drop([close_label], axis=1)
    df_X_matrix = df_X.to_numpy()
    x_train = df_X_matrix[:-prediction_length,:]
    x_test = df_X_matrix[-prediction_length:,:]
    
    df_Y = df[close_label]
    df_Y_matrix = df_Y.to_numpy()
    y_train = df_Y_matrix[:-prediction_length]
    y_test = df_Y_matrix[-prediction_length:]
    return x_train, y_train, x_test, y_test

def split_data_for_period(df, period = 50, steps=10):
    df_matrix = df.to_numpy()
    
    x_values = []
    y_values = []

    for i in range(period,len(df)-steps):
        x_values.append(df_matrix[i-period:i,0])
        y_values.append(df_matrix[i+steps,0])
        
    return np.array(x_values), np.array(y_values)

def split_data_for_period_V2(df, period = 50, steps=10):
    df_matrix = df.to_numpy()
    
    x_values = []
    y_values = []

    for i in range(len(df)):
        end_index = i + period
        output_index = end_index + steps
        if i < 10:
            print("i = {}".format(i))
            print("end_index = {}".format(end_index))
            print("output_index = {}".format(output_index))
        if output_index < len(df):
            x_values.append(df_matrix[i:end_index])
            y_values.append(df_matrix[output_index])
            
    return np.array(x_values), np.array(y_values)

def split_data_for_period_and_outsteps(df, period = 50, outsteps=10):
    df_matrix = df.to_numpy()
    
    x_values = []
    y_values = []

    for i in range(period,len(df)-outsteps):
        x_values.append(df_matrix[i-period:i,0])
        y_values.append(df_matrix[i:i+outsteps,0])
        
    return np.array(x_values), np.array(y_values)

def split_data_portion(df, prediction_portion, close_label = 'Adj Close'):
    prediction_length = int(len(df)*prediction_portion)
    return split_data(df, prediction_length, close_label = 'Adj Close')

def define_data_group(df, validation_start_index, test_start_index):
    train_df = df[:validation_start_index]
    val_df = df[validation_start_index:test_start_index]
    test_df = df[test_start_index:]
    return train_df, val_df, test_df

def split_and_standardize_data(df, validation_start_index, test_start_index):
    train_df, val_df, test_df = define_data_group(df, validation_start_index, test_start_index)

    df_mean = df.mean()
    df_std = df.std()

    train_df = (train_df - df_mean) / df_std
    val_df = (val_df - df_mean) / df_std
    test_df = (test_df - df_mean) / df_std
    return train_df, val_df, test_df

def scale_and_split(df, validation_start_index, test_start_index):
    data_norm = (df - df.min())/(df.max()-df.min())
    return data_norm

def split_data_for_steps(df, steps=5, close_label = 'Adj Close'):
    df_X = df.drop([close_label], axis=1)
    df_X_matrix = df_X.to_numpy()
    x_features = df_X_matrix[:-steps,:]
    
    df_Y = df[close_label]
    df_Y_matrix = df_Y.to_numpy()
    y_labels = df_Y_matrix[steps:]

    return x_features, y_labels