#import libraries
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
import datetime as dt

#function to filter covid data by country
def get_country_data(frame: pd.DataFrame, country_code: str) -> pd.DataFrame:
    """
    This function filters the covid data by specific country.
    :param frame: original data frame that needs to be filtered.
    :param country_code: country code to filter based on country.
    :return: filtered dataframe.
    """
    df = frame[frame['Country_code'] == country_code]
    df.drop(['WHO_region', 'New_deaths', 'Cumulative_deaths', 'Country'], axis=1, inplace=True)
    return df

#function to process dataframe
def process_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Function to preprocess dataframe for datetime and filter data for year 2020-2021.
    :param frame: original dataframe
    :return: frame with data for year 2020-21.
    """
    frame['Date'] = pd.to_datetime(frame['Date'])
    frame['month'] = frame['Date'].dt.month
    frame['year'] = frame['Date'].dt.year
    frame = frame[frame['year'].isin([2020, 2021])]
    frame['date']=frame['Date'].dt.strftime('%m-%Y')
    frame.reset_index(inplace =True,drop=True)
    return frame