# import libraries
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import datetime as dt
import calendar
import seaborn as sns

warnings.filterwarnings('ignore')


# function to filter covid data by country
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


# function to process dataframe
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
    frame['date'] = frame['Date'].dt.strftime('%m-%Y')
    frame.reset_index(inplace=True, drop=True)
    return frame


# plot to compare monthly energy consumption in pre covid and during covid scenarios.
def compare_monthly_energy_with_covid(frame: pd.DataFrame, covid_df: pd.DataFrame):
    """
    This function plots comparison between monthly energy consumption before and during pandemic in US.
    :param frame: Dataframe with monthly energy data
    :param covid_df: Dataframe with monthly covid data
    :return: plot
    """
    frame = frame.iloc[1:]
    frame.rename(columns={'Month': "Date"}, inplace=True)
    frame['Date'] = pd.to_datetime(frame['Date'])
    y = [2017, 2018, 2019, 2020, 2021]
    frame = process_dataframe(frame, y)
    frame = frame[['Date', 'Total Fossil Fuels Consumption', 'Total Renewable Energy Consumption',
                   'Total Primary Energy Consumption', 'year', 'month']]
    frame = pd.merge(frame, covid_df, on=['year', 'month'], how='left')
    frame.drop(columns={'year', 'month', 'Cumulative_cases'}, inplace=True)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(frame['New_cases'], label='no of cases', color='r')
    ax.legend(loc=1)
    plt.xticks(np.arange(0, 56, 1), frame['Date'].dt.strftime('%m-%Y'), rotation=60)
    plt.ylabel("Covid Cases (in Million)")
    plt.xlabel("Timeline")
    ax2 = ax.twinx()
    ax2.plot(frame['Total Primary Energy Consumption'], label="energy consumption")
    ax2.legend(loc=2)
    plt.ylabel("Energy Consumption(in Quadrillion Btu)")
    plt.title("Energy Consumption in US during pandemic")
    x = [12, 24, 36, 48]
    for i in x:
        plt.axvline(x=i, c='y')
    plt.show()


def process_electricity_consumption(total_electricity_consump: pd.DataFrame):
    """
    Process the dataframe with total electricity consumption around world and US.
    :param total_electricity_consump: dataframe containing total electricity data
    :return: processed dataframe with required columns for plotting total electricity consumption
    """
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


# calculate per person electricity consumption in US and World.
def per_person_electricity_consumption(population: pd.DataFrame, total_electricity_consump: pd.DataFrame):
    """
    This function plots the per capita electricity consumption around the world and US and it's changes over the years.
    :param population: contains data related to population.
    :param total_electricity_consump: dataframe from process_electricity_consumption
    :return: plots for per capita consumption over the years.
    """
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
    plt.xlabel("Year\n\n")
    plt.ylabel("Consumption(in Million kWh)")
    plt.title("Per person Electricity Consumption in World")
    plt.xticks(rotation=90)

    plt.subplot(2, 2, 3)
    plt.plot(population_world['Year'], population_world['Perperson Electricity Consumption US'], color='r')
    plt.xlabel("Year")
    plt.ylabel("Consumption(in Million kWh)")
    plt.title("Per person Electricity Consumption in US")
    plt.xticks(rotation=90)
    plt.show()


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


def change_to_datetime(df: DataFrame) -> None:
    """
    :param df: DataFrame of which name of columns will be compared to the list and
    data types will be changed
    :return: None
    """
    # To fix errors in pd.to_datetime reference was taken from ->
    # https://datascientyst.com/how-to-fix-pandas-to_datetime-wrong-date-and-errors/
    word_date = ['Date', 'date', 'Month', 'month', 'Year', 'year']  # creating list of words
    for column_name in [column for word in word_date for column in df.columns if word in column]:
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce')


######################
def give_covid_df_yearly(year, country, data: DataFrame):
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
# -----------------------------function for world energy trends-------------------------------------
# function to form a yearly data :
def get_yearly_df(sheet_name: str) -> DataFrame:
    """
    :param sheet_name: excel file sheet name of which analysis is to be done
    :return: filtered data frame of growth rate and rest of regions
    """
    # reading the file
    world_data = pd.read_excel('Data/bp-stats-review-2021-all-data.xlsx', sheet_name=sheet_name, skiprows=2, index_col=0)
    df_to_drop = world_data.loc['of which: OECD':, :]  # filtering unwanted rows
    world_data.drop(df_to_drop.index, inplace=True)  # dropping them
    world_data = world_data.loc[:, :'2020.1']  # dropping unwanted columns
    world_data.rename(columns={'2020.1': 'Growth Rate in 2020'}, inplace=True)
    growth_df = pd.DataFrame(world_data['Growth Rate in 2020'])  # converting the growth_rate column into data frame
    world_data.drop(columns=['Growth Rate in 2020'], inplace=True)  # Dropping the growth_rate column also
    return (world_data, growth_df)  # return a tuple


def sheet_to_dict(world_data_df: DataFrame, growth_rate: DataFrame) -> DataFrame:
    """
    The function will take the input from function get_yearly_df() and will also plot the line plot
    :param world_data_df: Data frame, first return value of get_yearly_df()
    :param growth_rate: Data frame, second return valur of get_yearly_df()
    :return: DataFrame, of the growth rate 2020
    """
    # Creating a dictionary for main data frame
    df_dict = {'Year': [], 'Total North America': [], 'Total S. & Cent. America': [],
               'Total Europe': [],
               'Total CIS': [],
               'Total Middle East': [],
               'Total Africa': [],
               'Total Asia Pacific': [],
               'Total World': []}
    # Creating a dictionary for growth rate data
    growth_rate_dict = {'Total North America': [],
                        'Total S. & Cent. America': [],
                        'Total Europe': [],
                        'Total CIS': [],
                        'Total Middle East': [],
                        'Total Africa': [],
                        'Total Asia Pacific': [],
                        'Total World': []}

    def round_series(r_name: str, df: DataFrame) -> list:
        """
        :param r_name: index name of the data frame
        :param df: data frame from which row is to be pulled
        :return: list, rounding each element from the array
        """
        # the row is pulled, each value is rounded and array is converted into list
        lst = [round(val, 2) for val in df.loc[r_name, :].to_list()]
        return lst

    # fetching the val of column name since they are years
    for year in [col for col in world_data_df.columns]:
        df_dict['Year'].append(int(year))  # converting into int
    for key in df_dict.keys():  # Traversing the keys of df_dict
        if key == 'Year':
            pass
        else:
            # Traversing only those index value in which has string 'Total'
            for row_name in [ind for ind in world_data_df.index if 'Total' in ind]:
                if row_name == key:
                    # calling the function round_series() which will return list
                    l = round_series(row_name, world_data_df)
                    df_dict[key] = l  # assigning  the list to appropriate key of dictionary
    # working with growth rate df
    for key in growth_rate_dict.keys():
        for row_name in [ind for ind in growth_rate.index if 'Total' in ind]:
            if row_name == key:
                l = round_series(row_name, growth_rate)  # converting array to a list
                growth_rate_dict[key] = [x * 100 for x in l]

    # converting both the dictionaries into data frame
    main_df = pd.DataFrame(data=df_dict)
    growth_df = pd.DataFrame(data=growth_rate_dict)

    return (main_df, growth_df)  # returning a tuple


# function to plot the graph
def plot_graph_file_name(filename: str, title: str) -> DataFrame:
    """
    The function will take the file name and Title to plot the line graph
    :param filename: The file name indicates the name of sheet name of excel file to analyse
    :param title: The title of the graph
    :return: Data Frame of growth rate in the year 2020
    """
    # Calling the function get_yearly_df to clean the data
    raw_df, raw_df_growth = get_yearly_df(filename)[0], get_yearly_df(filename)[1]
    # Calling the function to get the set of two data frames
    both_df = sheet_to_dict(raw_df, raw_df_growth)
    final_energy_data = both_df[0]  # fetching the cleaned data frame of energy from the set both_df
    final_energy_growth = both_df[1]  # fetching the cleaned data frame of %change in 2020
    print('Growth Rate - 2020')
    # Code for plotting the line graph
    plt.figure(figsize=(15, 8))  # Setting the figure size
    for col in [col for col in final_energy_data.columns if col != 'Year']:
        sns.set_theme(style="darkgrid")  # Setting the theme
        sns.lineplot(data=final_energy_data, x=[int(x) for x in final_energy_data['Year']],
                     y=final_energy_data[col], marker="o")  # plotting line graph
    plt.legend([col for col in final_energy_data.columns if col != 'Year'])  # Giving the legend
    plt.title(title, fontname="Times New Roman", size=20, fontweight='bold')
    plt.xlabel('Year', fontweight='bold', fontname="Times New Roman", size=15)
    plt.ylabel(filename, fontweight='bold', fontname="Times New Roman", size=15)
    return final_energy_growth


def percent_change(frame: pd.DataFrame, colname: str, newcolname: str) -> pd.DataFrame:
    frame[newcolname] = np.nan
    for row in range(1, len(frame[colname])):
        if (frame[colname][row] == '--'):
            frame[colname][row] = np.nan
        frame[newcolname][row] = ((float(frame[colname][row]) - float(frame[colname][row - 1])) / float(
            frame[colname][row])) * 100
    return frame


def plot_emission_vs_energy(col1, col2, col3, label1, label2, title):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(col1, col2.astype(float), color='r', label=label1)
    ax.legend(loc=1)
    plt.ylabel(label1)
    plt.xlabel("Timeline")
    plt.xticks(rotation=90)
    ax2 = ax.twinx()
    ax2.plot(col3.astype(float), color='b', label=label2)
    ax2.legend(loc=2)
    plt.ylabel(label2)
    plt.title(title)
    plt.show()


def plot_per_change_emission_vs_energy(col1, col2, col3, label, title, l1, l2):
    plt.figure(figsize=(12, 6))
    plt.plot(col1, col2, color='r')
    plt.plot(col1, col3, color='g')
    plt.legend(labels=[l1, l2])
    plt.xticks(rotation=90)
    plt.ylabel(label)
    plt.title(title)
    plt.show()


def prepare_emission_consumption(frame_emission, frame_consumption):
    frame_emission.reset_index(inplace=True)
    frame_emission = frame_emission.iloc[2:, 2:]
    frame_emission.rename(columns={'Unnamed: 1': "Emissions"}, inplace=True)
    frame_emission = frame_emission.set_index("Emissions")
    frame_emission = pd.DataFrame.transpose(frame_emission)
    frame_emission.reset_index(inplace=True)
    frame_emission.rename(columns={'index': "Year"}, inplace=True)
    for x in frame_emission.columns:
        frame_emission.rename(columns={x: x.strip()}, inplace=True)
    frame_emission = frame_emission.iloc[:, :2]
    frame_emission = percent_change(frame_emission, 'CO2 emissions (MMtonnes CO2)', '%change emission')
    frame_consumption.drop("API", axis=1, inplace=True)
    frame_consumption.rename(columns={"Unnamed: 1": "Geography"}, inplace=True)
    frame_consumption = frame_consumption.iloc[1:, :]
    frame_consumption.set_index('Geography', inplace=True)
    frame_consumption = pd.DataFrame.transpose(frame_consumption)
    frame_consumption.reset_index(inplace=True)
    frame_consumption.rename(columns={"index": "Year"}, inplace=True)
    for x in frame_consumption.columns:
        frame_consumption.rename(columns={x: x.lstrip()}, inplace=True)
    frame_consumption = frame_consumption.iloc[:, :6]
    frame_consumption['Total Fossil Fuel'] = frame_consumption['Coal (quad Btu)'].astype(float) + frame_consumption[
        'Natural gas (quad Btu)'].astype(float) + frame_consumption['Petroleum and other liquids (quad Btu)'].astype(
        float)
    frame_consumption = percent_change(frame_consumption, 'Total Fossil Fuel', '%change Fossil Fuel')
    frame_consumption = percent_change(frame_consumption, 'Nuclear, renewables, and other (quad Btu)',
                                       '%change Renewable')
    frame_consumption = percent_change(frame_consumption, 'Consumption (quad Btu)', '%change total consumption')
    frame_emission = pd.merge(frame_emission, frame_consumption, on='Year', how='inner')
    frame_emission.iloc[0, 2] = np.nan
    plot_emission_vs_energy(frame_emission['Year'], frame_emission['CO2 emissions (MMtonnes CO2)'].astype(float),
                            frame_emission['Consumption (quad Btu)'], "CO2 Emissions (in MMtonnes CO2)",
                            'Energy Consumption(in Quadrillion Btu)',
                            'Emissions vs Total Energy Consumptions over the years')
    plot_emission_vs_energy(frame_emission['Year'], frame_emission['CO2 emissions (MMtonnes CO2)'].astype(float),
                            frame_emission['Total Fossil Fuel'].astype(float), "CO2 Emissions (in MMtonnes CO2)",
                            ' Fossil Energy Consumption(in Quadrillion Btu)',
                            'Emissions vs Fossil Energy Consumptions over the years')
    plot_emission_vs_energy(frame_emission['Year'], frame_emission['CO2 emissions (MMtonnes CO2)'].astype(float),
                            frame_emission['Nuclear, renewables, and other (quad Btu)'].astype(float),
                            "CO2 Emissions (in MMtonnes CO2)", ' Renewable Energy Consumption(in Quadrillion Btu)',
                            'Emissions vs Renewable Energy Consumptions over the years')
    plot_per_change_emission_vs_energy(frame_emission['Year'], frame_emission['%change emission'],
                                       frame_emission['%change total consumption'], 'Percentage change',
                                       ' Percent change in total emission vs total consumption over years',
                                       'CO2 emission', "Total Consumption")
    plot_per_change_emission_vs_energy(frame_emission['Year'], frame_emission['%change emission'],
                                       frame_emission['%change Fossil Fuel'], 'Percentage change',
                                       ' Percent change in total emission vs total fossil fuel  energy consumption over years',
                                       'CO2 emission', "Total  Fossil fuel Consumption")
    plot_per_change_emission_vs_energy(frame_emission['Year'], frame_emission['%change emission'],
                                       frame_emission['%change Renewable'], 'Percentage change',
                                       ' Percent change in total emission vs total renewable energy consumption over years',
                                       'CO2 emission', "Total Renewable Consumption")

