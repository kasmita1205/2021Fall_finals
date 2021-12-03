#import libraries
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
import datetime as dt
import calendar

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


def process_electricity_consumption(total_electricity_consump: pd.DataFrame):
    total_electricity_consump.reset_index(inplace=True, drop=True)
    total_electricity_consump.drop("API", axis=1, inplace=True)
    total_electricity_consump.rename(columns={"Unnamed: 1": "Geography"}, inplace=True)
    total_electricity_consump = total_electricity_consump.iloc[1:, :]
    total_electricity_consump = total_electricity_consump.set_index("Geography")
    total_electricity_consump = pd.DataFrame.transpose(total_electricity_consump)
    total_electricity_consump.reset_index(inplace=True)
    total_electricity_consump.rename(columns={"index": "Year", "Geography": "index"}, inplace=True)
    for x in total_electricity_consump.columns:
        total_electricity_consump.rename(columns={x: x.lstrip()}, inplace=True)
    total_electricity_consump = total_electricity_consump[['Year', 'World', 'United States']]
    return total_electricity_consump


def per_person_electricity_consumption(population: pd.DataFrame, total_electricity_consump: pd.DataFrame):
    population = population.iloc[1:, 1:]
    population.rename(columns={'Unnamed: 1': "Geography"}, inplace=True)
    population = population.set_index("Geography")
    population = pd.DataFrame.transpose(population)
    population.reset_index(inplace=True)
    population.rename(columns={"index": "Year", "Geography": "index"}, inplace=True)
    for x in population.columns:
        population.rename(columns={x: x.lstrip()}, inplace=True)
    population_world = population[['Year', 'World', 'United States']]
    population_world = pd.merge(population_world, total_electricity_consump, on='Year', how='inner')
    population_world.rename(columns={"World_x": "World Population", "United States_x": "US Population",
                                     'World_y': "World Electricity Consumption",
                                     "United States_y": "US Electricity Consumption"}, inplace=True)
    population_world = population_world[
        ["Year", "World Population", "US Population", "World Electricity Consumption", "US Electricity Consumption"]]
    population_world["Perperson Electricity Consumption World"] = (population_world[
                                                                       'World Electricity Consumption'].astype(float) /
                                                                   population_world["World Population"].astype(
                                                                       float)) * 1000
    population_world["Perperson Electricity Consumption US"] = (population_world['US Electricity Consumption'].astype(
        float) / population_world["US Population"].astype(float)) * 1000
    plt.figure(figsize=[20, 10])
    plt.subplot(2, 2, 1)
    plt.plot(population_world['Year'], population_world['Perperson Electricity Consumption World'])
    plt.xlabel("Year")
    plt.ylabel("Consumption(in Million kWh)")
    plt.title("Perperson Energy Consumption in World")
    plt.xticks(rotation=90)

    plt.subplot(2, 2, 3)
    plt.plot(population_world['Year'], population_world['Perperson Electricity Consumption US'], color='r')
    plt.xlabel("Year")
    plt.ylabel("Consumption(in Million kWh)")
    plt.title("Perperson Energy Consumption in US")
    plt.xticks(rotation=90)
    plt.show()


# ------------------------------------------------------------------------------------
# function to clean fuel_price_df
def clean_fuel_price(df):
    fuel_price = df.T.copy()
    fuel_price.reset_index(inplace=True)
    fuel_price.columns = fuel_price.iloc[0]
    fuel_price = fuel_price.iloc[1:, :]
    fuel_price.rename({'Monthly': 'Date/Time'}, axis=1, inplace=True)
    #####
    change_to_datetime(fuel_price)
    fuel_price = fuel_price[fuel_price['Date/Time'].dt.year > 2018]
    fuel_price = fuel_price[['Date/Time', 'OPEC - ORB']]
    return fuel_price


def plot_null_values(df):
    '''This function give a rough idea about the presence of null values in the dataframe
    '''
    print('The shape of data frame is = {}'.format(df.shape))
    plt.subplots(figsize=(20, 5))
    sns.heatmap(df.isna(), yticklabels=False)
    plt.xlabel('Column Name', fontsize=15)
    plt.title('Missing value plot', fontsize=15)

    plt.show()


def change_to_datetime(df):
    '''This function will check the column names of the data frame df and then will compare it with the words provided
    in word list, if any of the word present in column name matches with the word in the list the data type of the column
    will be changed to datetime format.
    '''
    word_date = ['Date', 'date', 'Month', 'month', 'Year', 'year']
    for column_name in [column for word in word_date for column in df.columns if word in column]:
        # print(column_name)
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce')


######################
def give_covid_df_yearly(year, country, data):
    df = data[['Date_reported', 'Country', 'New_cases']]  # selecting required cols
    df_filtered = pd.DataFrame()
    if country != 'all':
        df_filtered = df.loc[(df['Date_reported'].dt.year == year) & (df['Country'] == country)]
    if country == 'all':
        df_filtered = df[df['Date_reported'].dt.year == year]

    df_grouped = df_filtered.groupby(df_filtered['Date_reported'].dt.month)['New_cases'].sum().reset_index()
    df_grouped['Date_reported'] = df_grouped['Date_reported'].apply(lambda x: calendar.month_abbr[x])
    df_grouped['Date_reported'] = df_grouped['Date_reported'] + '-' + str(year)
    change_to_datetime(df_grouped)

    return df_grouped


def get_monthly_covid_df(country_name, dataf):
    df_2020 = give_covid_df_yearly(2020, country_name, dataf)
    df_2021 = give_covid_df_yearly(2021, country_name, dataf)
    monthly_covid_new_cases = pd.concat([df_2020, df_2021], ignore_index=True)
    monthly_covid_new_cases.rename(columns={'Date_reported': 'Date/Time'}, inplace=True)
    return monthly_covid_new_cases


#############################

def energy_plot_func(energy_source_1, energy_source_2, df):
    energy_US = df
    plt.figure(figsize=(15, 10))
    graph1 = sns.lineplot(x='Month', y=energy_source_1, data=energy_US)
    graph2 = sns.lineplot(x='Month', y=energy_source_2, data=energy_US)
    plt.legend([energy_source_1, energy_source_2])
    plt.title(energy_source_1 + ' and ' + energy_source_2 + ' analysis')
    plt.show()

# --------------------------------------------------------------------------------------------------

