# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:31:12 2023

@author: diyaa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def GetData(job):
    job_df = pd.read_csv(job)
    return job_df

job_df_melted = GetData('job.csv').melt(id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'], var_name='Year', value_name='Value')

# Assuming melted_df is your df
# Replace non-numeric values with NaN
job_df_melted.replace('..', np.nan, inplace=True)
job_df_pivoted = job_df_melted.groupby(['Country Name', 'Country Code', 'Year', 'Series Name'])['Value'].mean().unstack().reset_index()

# Creating a pivot table from pivoted_df
job_df_countries = job_df_melted.pivot_table(index=['Year', 'Series Name'], columns='Country Name', values='Value')
job_df_years = job_df_melted.pivot_table(index=['Country Name', 'Series Name'], columns='Year', values='Value')

job_df_cleaned = job_df_pivoted.fillna(job_df_pivoted.mean())

# Applying Statistical Methods on cleaned dataset
job_df_cleaned_new = job_df_cleaned.drop(['Year', 'Country Name'], axis='columns')
print(job_df_cleaned_new.describe())

def Employement_Bar_Graph(Job_Data):
    # Read the data from the CSV file
    emp_data = pd.read_csv(Job_Data)

    # Filter data for the specified countries and years
    countries = ['Pakistan', 'Germany',
                 'United Kingdom', 'Egypt, Arab Rep.', 'Spain']
    
    years = range(2000, 2009)
    
    filtered_emp_data = emp_data[(emp_data['Country Name'].isin(
        countries)) & (emp_data['Year'].isin(years))]

    # Pivot the data for grouped bar chart
    pivoted_emp_data = filtered_emp_data.pivot_table(
        index='Country Name', columns='Year', values='Total employment, total (ages 15+)')

    # Plotting the grouped bar chart
    pivoted_emp_data.plot(kind='bar', figsize=(10, 6))
    plt.title('Employment Rate of countries')
    plt.legend(title='Years')
    plt.xlabel('Countries')
    plt.ylabel('Employment Rate')
    plt.legend(title='Years')
    plt.grid(True)
    plt.savefig('employement_bar_graph', dpi=300)
    plt.show()  
Employement_Bar_Graph('job_df_cleaned.csv')

def Population_Bar_Graph(Job_Data):
    # Read the data from the CSV file
    pop_data = pd.read_csv(Job_Data)

    # Filter data for the specified countries and years
    countries = ['Pakistan', 'Germany',
                 'United Kingdom', 'Egypt, Arab Rep.', 'Spain']
    years = range(2000, 2009)
    filtered_pop_data = pop_data[(pop_data['Country Name'].isin(
        countries)) & (pop_data['Year'].isin(years))]

    # Pivot the data for grouped bar chart
    pivoted_pop_data = filtered_pop_data.pivot_table(
        index='Country Name', columns='Year', values='Population, total')

    # Plotting the grouped bar chart
    pivoted_pop_data.plot(kind='bar', figsize=(10, 6))
    plt.title('Population of countries')
    plt.xlabel('Country')
    plt.ylabel('Population')
    plt.grid(True)
    plt.legend(title='Year')
    plt.savefig('job_df_cleaned', dpi=300)
    plt.show()
Population_Bar_Graph('job_df_cleaned.csv')

def Gdp_Line_Plot(Job_Data):
    # Load the dataset from the CSV file
    gdp_data = pd.read_csv(Job_Data)

    # Filter data for the specified countries and years
    countries = ['Pakistan', 'Germany', 'United Kingdom', 'Egypt, Arab Rep.', 'Spain']
    years = list(range(2000, 2009))

    filtered_gdp_data = gdp_data[(gdp_data['Country Name'].isin(countries)) & (gdp_data['Year'].isin(years))]

    # Pivot the data for easier plotting
    pivoted_gdp_data = filtered_gdp_data.pivot(index='Year', columns='Country Name', values='GDP growth (annual %)')

    # Plotting
    plt.figure(figsize=(10, 6))

    for country in countries:
        plt.plot(pivoted_gdp_data.index, pivoted_gdp_data[country], label=country)

    plt.title('GDP of Countries')
    plt.xlabel('Year')
    plt.ylabel('GDP')
    plt.legend()
    plt.grid(True)
    plt.show()
Gdp_Line_Plot('job_df_cleaned.csv')

def Tax_Line_Plot(Job_Data):
    # Load the dataset from the CSV file
    tax_data = pd.read_csv(Job_Data)

    # Filter data for the specified countries and years
    countries = ['Pakistan', 'Germany', 'United Kingdom', 'Egypt, Arab Rep.', 'Spain']
    years = list(range(2000, 2009))

    filtered_tax_data = tax_data[(tax_data['Country Name'].isin(countries)) & (tax_data['Year'].isin(years))]

    # Pivot the data for easier plotting
    pivoted_tax_data = filtered_tax_data.pivot(index='Year', columns='Country Name', values='Tax revenue (% of GDP)')

    # Plotting
    plt.figure(figsize=(10, 6))
 
    for country in countries:
        plt.plot(pivoted_tax_data.index, pivoted_tax_data[country], label=country)

    plt.title('Tax Revenue')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    plt.show()
Tax_Line_Plot('job_df_cleaned.csv')

def Germany_HeatMap(Job_Data):
    Heat_data = pd.read_csv(Job_Data)
    Germany_data = Heat_data[Heat_data['Country Name'] == 'Germany']

# Select relevant indicators
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

# Create a subset of data with selected 
    Germany_subset = Germany_data[indicators]

# Plotting the heatmap
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(Germany_subset.corr(), annot=True, cmap='magma', fmt='.2f', annot_kws={"size": 10})
    heatmap.set_xticklabels([label[:20] + '...' if len(label) > 20 else label for label in Germany_subset.columns])
    heatmap.set_yticklabels([label[:20] + '...' if len(label) > 20 else label for label in Germany_subset.columns])
    plt.title('Correlation Heatmap of Indicators for Germany')
    plt.show()
Germany_HeatMap('job_df_cleaned.csv')

def Egypt_HeatMap(Job_Data):
    Heat_data = pd.read_csv(Job_Data)
    Egypt_data = Heat_data[Heat_data['Country Name'] == 'Egypt, Arab Rep.']

# Select relevant indicators
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

# Create a subset of data with selected indicators
    Egypt_subset = Egypt_data[indicators]

# Plotting the heatmap
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(Egypt_subset.corr(), annot=True, cmap='ocean', fmt='.2f', annot_kws={"size": 10})
    heatmap.set_xticklabels([label[:20] + '...' if len(label) > 20 else label for label in Egypt_subset.columns])
    heatmap.set_yticklabels([label[:20] + '...' if len(label) > 20 else label for label in Egypt_subset.columns])
    plt.title('Correlation Heatmap of Indicators for Egypt')
    plt.show()
Egypt_HeatMap('job_df_cleaned.csv')

def United_Kingdom_HeatMap(Job_Data):
    Heat_data = pd.read_csv(Job_Data)
    UK_data = Heat_data[Heat_data['Country Name'] == 'United Kingdom']

# Select relevant indicators
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

# Create a subset of data with selected indicators
    UK_subset = UK_data[indicators]

# Plotting the heatmap
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(UK_subset.corr(), annot=True, cmap='twilight', fmt='.2f', annot_kws={"size": 10})
    heatmap.set_xticklabels([label[:20] + '...' if len(label) > 20 else label for label in UK_subset.columns])
    heatmap.set_yticklabels([label[:20] + '...' if len(label) > 20 else label for label in UK_subset.columns])
    plt.title('Correlation Heatmap of Indicators for United Kingdom')
    plt.show()
United_Kingdom_HeatMap('job_df_cleaned.csv')



