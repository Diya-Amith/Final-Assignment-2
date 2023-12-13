# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:31:12 2023

@author: diyaa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis
import seaborn as sns

def GetData(job): 
    """
    Read the dataset from the csv file    
    Returns the loaded Dataframe containing the data from the csv file. 
    """
    job_df = pd.read_csv(job)
    return job_df

# Load data and melt it for easier manipulation
job_df_melted = GetData('job.csv').melt(id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'], var_name='Year', value_name='Value')

# Replace non-numeric values with NaN for easier processing
job_df_melted.replace('..', np.nan, inplace=True)

# Group melted data to create a pivoted DataFrame
job_df_pivoted = job_df_melted.groupby(['Country Name', 'Country Code', 'Year', 'Series Name'])['Value'].mean().unstack().reset_index()

# Creating a pivot table from pivoted_df
job_df_countries = job_df_melted.pivot_table(index=['Year', 'Series Name'], columns='Country Name', values='Value')
job_df_years = job_df_melted.pivot_table(index=['Country Name', 'Series Name'], columns='Year', values='Value')

# Fill missing values with the mean of each column
job_df_cleaned = job_df_pivoted.fillna(job_df_pivoted.mean())

# Applying Statistical Methods on cleaned dataset
job_df_cleaned_new = job_df_cleaned.drop(['Year', 'Country Name'], axis='columns')

#describe method()
df_describe = job_df_cleaned_new.describe()
print(df_describe)

#skewness
population_Germany= job_df_cleaned[job_df_cleaned['Country Name'] == "Germany"]
Germany_Skewness = skew(population_Germany["Population, total"])
print(f"skewnwss of Germany population data {Germany_Skewness}") 

#kurtosis
Germany_Kurtosis = kurtosis(population_Germany["Population, total"])
print(f"kurtosis of Germany population data{Germany_Kurtosis}")

# Bar graph for employement rate in chosen countries
def Employement_Bar_Graph(Job_Data):
    
    """    
    Visualizes and compares the employment rates of specific countries over a set of years.    
    Reads the data from the CSV file or DataFrame    
    Filters the dataset based on specific countries and years from 2000 t0 2008    
    Creates a pivot table to structure the data for plotting a bar chart.
    Plots a bar chart to display the employment rates for selected countries across different years.
    Generates a bar chart visualizing the employment rates of the chosen countries.    
    Saves the plot as an image file named "Employement_Bar_Graph.png".    
    """
    
    # Read the data from the CSV file
    emp_data = pd.read_csv(Job_Data)

    # Filter countries to plot
    countries = ['Pakistan', 'Germany',
                 'United Kingdom', 'Egypt, Arab Rep.', 'Spain']
    
    # Filter years to plot
    years = range(2000, 2009)
    
    # Filter data for the specified countries and years
    filtered_emp_data = emp_data[(emp_data['Country Name'].isin(
        countries)) & (emp_data['Year'].isin(years))]

    # Pivot the data for bar chart
    pivoted_emp_data = filtered_emp_data.pivot_table(
        index='Country Name', columns='Year', values='Total employment, total (ages 15+)')

    # Plot the bar chart
    pivoted_emp_data.plot(kind='bar', figsize=(10, 6))
    plt.title('Employment Rate of countries')
    
    # To plot legend and label for title, x axis and y axis
    plt.legend(title='Years')
    plt.title('Employement Rate of Countries')
    plt.xlabel('Countries')
    plt.ylabel('Employment Rate')    
    plt.grid(True)
    
    # Save the plot as an image and display it
    plt.savefig('Employement_Bar_Graph', dpi=300)
    plt.show()  
Employement_Bar_Graph('job_df_cleaned.csv')

# Bar graph for employement rate in chosen countries
def Population_Bar_Graph(Job_Data):
    
    """    
    Visualizes and compares the population rates of specific countries over a set of years.    
    Reads the data from the CSV file or DataFrame    
    Filters the dataset based on specific countries and years from 2000 t0 2008    
    Creates a pivot table to structure the data for plotting a bar chart.
    Plots a bar chart to display the population rates for selected countries across different years.
    Generates a bar chart visualizing the employment rates of the chosen countries.    
    Saves the plot as an image file named "Population_Bar_Graph.png".    
    """
    
    # Read the data from the CSV file
    pop_data = pd.read_csv(Job_Data)

    # Filter countries to plot
    countries = ['Pakistan', 'Germany',
                 'United Kingdom', 'Egypt, Arab Rep.', 'Spain']
    
    # Filter Years to plot
    years = range(2000, 2009)
    
    # Filter data for the specified countries and years
    filtered_pop_data = pop_data[(pop_data['Country Name'].isin(
        countries)) & (pop_data['Year'].isin(years))]

    # Pivot the data for bar chart
    pivoted_pop_data = filtered_pop_data.pivot_table(
        index='Country Name', columns='Year', values='Population, total')

    # Plot the bar chart
    pivoted_pop_data.plot(kind='bar', figsize=(10, 6))
    
    # To plot legend and label for title, x axis and y axis
    plt.legend(title='Year')
    plt.title('Population of countries')
    plt.xlabel('Country')
    plt.ylabel('Population')
    plt.grid(True)
    
    # Save the plot as an image and display it
    plt.savefig('Population_Bar_Graph', dpi=300)
    plt.show()
Population_Bar_Graph('job_df_cleaned.csv')

# Line Plot for GDP rate in chosen countries
def Gdp_Line_Plot(Job_Data):
    
    """    
    Visualizes and compares the GDP rates of specific countries over a set of years.    
    Reads the data from the CSV file or DataFrame    
    Filters the dataset based on specific countries and years from 2000 t0 2008    
    Creates a pivot table to structure the data for plotting a line plot.
    Plots a line plot to display the employment rates for selected countries across different years.
    Generates a line plot visualizing the GDP rates of the chosen countries.    
    Saves the plot as an image file named "Gdp_Line_Plot.png".    
    """
    
    # Read the data from the CSV file
    gdp_data = pd.read_csv(Job_Data)

    # Filter Countries to plot
    countries = ['Pakistan', 'Germany', 'United Kingdom', 'Egypt, Arab Rep.', 'Spain']
    
    # Filter Years to plot
    years = list(range(2000, 2009))

    # Filter data for the specified countries and years
    filtered_gdp_data = gdp_data[(gdp_data['Country Name'].isin(countries)) & (gdp_data['Year'].isin(years))]

    # Pivot the data for line chart
    pivoted_gdp_data = filtered_gdp_data.pivot(index='Year', columns='Country Name', values='GDP growth (annual %)')

    # Plot the line chart
    plt.figure(figsize=(10, 6))
    for country in countries:
        plt.plot(pivoted_gdp_data.index, pivoted_gdp_data[country], label=country)

    # To plot legend and label for title, x axis and y axis
    plt.legend()
    plt.title('GDP of Countries')
    plt.xlabel('Year')
    plt.ylabel('GDP')
    plt.grid(True)
    
    # To save and show the plot
    plt.savefig('Gdp_Line_Plot', dpi=300)
    plt.show()
Gdp_Line_Plot('job_df_cleaned.csv')

# Line Plot for tax revenue in chosen countries
def Tax_Line_Plot(Job_Data):
    
    """    
    Visualizes and compares the Tax rates of specific countries over a set of years.    
    Reads the data from the CSV file or DataFrame    
    Filters the dataset based on specific countries and years from 2000 t0 2008    
    Creates a pivot table to structure the data for plotting a line plot.
    Plots a line plot to display the employment rates for selected countries across different years.
    Generates a line plot visualizing the Tax rates of the chosen countries.    
    Saves the plot as an image file named "Tax_Line_Plot.png".    
    """
    
    
    # Read the data from the CSV file
    tax_data = pd.read_csv(Job_Data)

    # Filter Countries to plot
    countries = ['Pakistan', 'Germany', 'United Kingdom', 'Egypt, Arab Rep.', 'Spain']
    
    # Filter Years to plot
    years = list(range(2000, 2009))
        
    # Filter data for the specified countries and years
    filtered_tax_data = tax_data[(tax_data['Country Name'].isin(countries)) & (tax_data['Year'].isin(years))]

    # Pivot the data for line chart
    pivoted_tax_data = filtered_tax_data.pivot(index='Year', columns='Country Name', values='Tax revenue (% of GDP)')

    # Plot the line chart
    plt.figure(figsize=(10, 6))
    for country in countries:
        plt.plot(pivoted_tax_data.index, pivoted_tax_data[country], label=country)

    # To plot legend and label for title, x axis and y axis
    plt.title('Tax Revenue')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.savefig('Tax_Line_Plot', dpi=300)
    plt.show()
Tax_Line_Plot('job_df_cleaned.csv')

# Heat Map for Germany
def Germany_HeatMap(Job_Data):
    
    """
    Creates a correlation heatmap for various economic indicators specifically for Germany. 
    Reads the data from the CSV file or DataFrame.
    Selects the data specifically for Germany 
    Filters out indicators related to population, tax revenue, employment, ATMs, employers, GDP growth, and labor force.
    Creates a subset of economic indicators for Germany.
    Generates a correlation heatmap using seaborn (sns) for the selected indicators.
    """
    
    # Read the data from the CSV file
    Heat_data = pd.read_csv(Job_Data)
    
    # Select the country
    Germany_data = Heat_data[Heat_data['Country Name'] == 'Germany']

    # Filter Indicators to plot
    indicators = [
       'Population, total',
       'Tax revenue (% of GDP)',
       'Total employment, total (ages 15+)',
       'Automated teller machines (ATMs) (per 100,000 adults)',
       'Employers, total (% of total employment) (modeled ILO estimate)',
       'Employment in agriculture (% of total employment) (modeled ILO estimate)',
       'Employment in industry (% of total employment) (modeled ILO estimate)',
       'GDP growth (annual %)',
       'Labor force, total'
         ]

    # Create a subset of country
    Germany_subset = Germany_data[indicators]

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(Germany_subset.corr(), annot=True, cmap='magma', fmt='.2f', annot_kws={"size": 10})
    
    # To truncate names of indicators in x axis
    heatmap.set_xticklabels([label[:20] + '...' if len(label) > 20 else label for label in Germany_subset.columns])
    
    # To truncate names of indicators in y axis
    heatmap.set_yticklabels([label[:20] + '...' if len(label) > 20 else label for label in Germany_subset.columns])
    
    # Give a title and show the plot
    plt.title('Correlation Heatmap of Indicators for Germany')
    plt.savefig('Germany_HeatMap', dpi=300)
    plt.show()
Germany_HeatMap('job_df_cleaned.csv')

# Heat Map for Egypt
def Egypt_HeatMap(Job_Data):
    
    """
    Creates a correlation heatmap for various economic indicators specifically for Egypt. 
    Reads the data from the CSV file or DataFrame.
    Selects the data specifically for Egypt 
    Filters out indicators related to population, tax revenue, employment, ATMs, employers, GDP growth, and labor force.
    Creates a subset of economic indicators for Egypt.
    Generates a correlation heatmap using seaborn (sns) for the selected indicators.
    """
    
    # Read the data from the CSV file
    Heat_data = pd.read_csv(Job_Data)
    
    # Select the country
    Egypt_data = Heat_data[Heat_data['Country Name'] == 'Egypt, Arab Rep.']

    # Filter Indicators to plot
    indicators = [
       'Population, total',
       'Tax revenue (% of GDP)',
       'Total employment, total (ages 15+)',
       'Automated teller machines (ATMs) (per 100,000 adults)',
       'Employers, total (% of total employment) (modeled ILO estimate)',
       'Employment in agriculture (% of total employment) (modeled ILO estimate)',
       'Employment in industry (% of total employment) (modeled ILO estimate)',
       'GDP growth (annual %)',
       'Labor force, total'
]

    # Create a subset of country
    Egypt_subset = Egypt_data[indicators]

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(Egypt_subset.corr(), annot=True, cmap='ocean', fmt='.2f', annot_kws={"size": 10})
    
    # To truncate names of indicators in x axis
    heatmap.set_xticklabels([label[:20] + '...' if len(label) > 20 else label for label in Egypt_subset.columns])
    
    # To truncate names of indicators in y axis
    heatmap.set_yticklabels([label[:20] + '...' if len(label) > 20 else label for label in Egypt_subset.columns])
    
    # Give a title and show the plot
    plt.title('Correlation Heatmap of Indicators for Egypt')
    plt.savefig('Egypt_HeatMap', dpi=300)
    plt.show()
Egypt_HeatMap('job_df_cleaned.csv')

# Heat Map for Egypt
def Pakistan_HeatMap(Job_Data):
    
    """
    Creates a correlation heatmap for various economic indicators specifically for Pakistan. 
    Reads the data from the CSV file or DataFrame.
    Selects the data specifically for Pakistan 
    Filters out indicators related to population, tax revenue, employment, ATMs, employers, GDP growth, and labor force.
    Creates a subset of economic indicators for pakistan.
    Generates a correlation heatmap using seaborn (sns) for the selected indicators.
    """
    
    # Read the data from the CSV file
    Heat_data = pd.read_csv(Job_Data)
    
    # Select the country
    Pakistan_data = Heat_data[Heat_data['Country Name'] == 'Egypt, Arab Rep.']

    # Filter Indicators to plot
    indicators = [
       'Population, total',
       'Tax revenue (% of GDP)',
       'Total employment, total (ages 15+)',
       'Automated teller machines (ATMs) (per 100,000 adults)',
       'Employers, total (% of total employment) (modeled ILO estimate)',
       'Employment in agriculture (% of total employment) (modeled ILO estimate)',
       'Employment in industry (% of total employment) (modeled ILO estimate)',
       'GDP growth (annual %)',
       'Labor force, total'
]

    # Create a subset of country
    Pakistan_subset = Pakistan_data[indicators]

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(Pakistan_subset.corr(), annot=True, cmap='twilight', fmt='.2f', annot_kws={"size": 10})
    
    # To truncate names of indicators in x axis
    heatmap.set_xticklabels([label[:20] + '...' if len(label) > 20 else label for label in Pakistan_subset.columns])
    
    # To truncate names of indicators in y axis
    heatmap.set_yticklabels([label[:20] + '...' if len(label) > 20 else label for label in Pakistan_subset.columns])
    
    # Give a title and show the plot
    plt.title('Correlation Heatmap of Indicators for Pakistan')
    plt.savefig('Pakistan_HeatMap', dpi=300)
    plt.show()
Pakistan_HeatMap('job_df_cleaned.csv')





