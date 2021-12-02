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
def process_dataframe(frame: pd.DataFrame, year) -> pd.DataFrame:
    """
    Function to preprocess dataframe for datetime and filter data for year 2020-2021.
    :param frame: original dataframe
    :param year: list of year
    :return: frame with data for year 2020-21.
    """
    frame['Date'] = pd.to_datetime(frame['Date'])
    frame['month'] = frame['Date'].dt.month
    frame['year'] = frame['Date'].dt.year
    frame = frame[frame['year'].isin(year)]
    frame['date']=frame['Date'].dt.strftime('%m-%Y')
    frame.reset_index(inplace =True,drop=True)
    return frame

def compare_monthly_energy_with_covid(frame: pd.DataFrame, covid_df :pd.DataFrame):
    """

    :param frame: Dataframe with monthly energy data
    :param covid_df: Dataframe with monthly covid data
    :return: plot
    """
    frame = frame.iloc[1:]
    frame.rename(columns = {'Month':"Date"}, inplace =True)
    frame['Date'] = pd.to_datetime(frame['Date'])
    y=[2017,2018,2019,2020,2021]
    frame = process_dataframe(frame, y)
    frame =frame[['Date', 'Total Fossil Fuels Consumption','Total Renewable Energy Consumption','Total Primary Energy Consumption', 'year','month']]
    frame =pd.merge(frame, covid_df, on=['year','month'], how='left')
    frame.drop(columns= {'year','month','Cumulative_cases'}, inplace =True)
    fig, ax = plt.subplots(figsize = (15,6))
    ax.plot(frame['New_cases'], label='no of cases', color ='r')
    ax.legend(loc=1)
    plt.xticks(np.arange(0,56,1), frame['Date'].dt.strftime('%m-%Y'), rotation=60)
    plt.ylabel("Covid Cases (in Million)")
    plt.xlabel("Timeline")
    ax2 = ax.twinx()
    ax2.plot(frame['Total Primary Energy Consumption'], label="energy consumption")
    ax2.legend(loc=2)
    plt.ylabel("Energy Consumption(in Quadrillion Btu)")
    plt.title("Energy Consumption in US during pandemic")
    x=[12,24,36,48]
    for i in x:
        plt.axvline(x=i, c='y')
    plt.show()