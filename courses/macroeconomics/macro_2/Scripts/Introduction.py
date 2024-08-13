#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################################
#  Introduction to Macroeconomics 2
##################################################

import pandas as pd
import requests
import seaborn as sns
import os
import re
import matplotlib.pyplot as plt


def year_trimester_to_date(year_trimester):
    """
    Converts a year-trimester string to a datetime object representing the start of the trimester.

    Parameters:
        year_trimester (str): A string in the format "YYYY-T#" where YYYY is the year and # is the trimester number.

    Returns:
        datetime: A datetime object representing the start of the trimester.

    Raises:
        ValueError: If the year_trimester string is not in the correct format.

    Example:
        >>> year_trimester_to_date("2022-T3")
        datetime.datetime(2022, 7, 1, 0, 0)
    """
    year, trimester = year_trimester.split('-T')
    month = (int(trimester) - 1) * 3 + 1  # Convert trimester to starting month
    return pd.to_datetime(f"{year}-{month:02d}-01")

def plot_pib_with_recessions(df, variable='PIB', variable_label='Changement du PIB (%)', display_mean = False, color='black',outpath=None):
    """
    Plot the PIB series with vertical grey blocks indicating recessions.

    Parameters:
        df (pandas.DataFrame): DataFrame with 'Date', 'PIB', and 'Recessions' columns.
        variable (str): Variable to plot. Defaults to 'PIB'.
        variable_label (str): Label for the variable. Defaults to 'Changement du PIB (%)'.
        display_mean (bool): Whether to display the mean PIB. Defaults to False.
        color (str): Color of the PIB line. Defaults to 'black'.
        outpath (str): Path to save the plot. If None, the plot will not be saved.

    Returns:
        None
    """
    
    import matplotlib.dates as mdates

    sns.set_style('white')
    # The recession data begins in 1960
    pib = df.copy()
    pib.dropna(subset=['Recessions'], inplace=True)
    pib.dropna(subset=[variable], inplace=True)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot the PIB series with vertical grey blocks indicating Recessions
    sns.lineplot(data=pib, x='Date', y=variable, color=color, ax=ax)
    ax.set(xlabel='Date', ylabel=variable_label)

    # Highlight Recessions periods
    recession_starts = pib.index[pib['Recessions'].eq(1) & pib['Recessions'].shift(1).ne(1)].tolist()
    recession_ends = pib.index[pib['Recessions'].eq(1) & pib['Recessions'].shift(-1).ne(1)].tolist()

    for start, end in zip(recession_starts, recession_ends):
        ax.axvspan(pib['Date'][start], pib['Date'][end], color=color, alpha=0.3)

    if display_mean:
        ax.axhline(y=pib[variable].mean(), color='black', linestyle='--', label='Moyenne')

    # Set up the x-axis to show more years and add ticks
    years = mdates.YearLocator(5)  # Tick every 5 years
    years_fmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)

    # Add minor ticks for each year
    ax.xaxis.set_minor_locator(mdates.YearLocator())

    # Format the x-axis
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Ensure the x-axis limits cover the full range of data
    ax.set_xlim(pib['Date'].min(), pib['Date'].max())

    # Add gridlines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close(fig)


def trimestre_to_quarter(trimestre):
    """
    Convert a trimester string to a quarter string.

    Parameters:
        trimestre (str): The trimester string in the format "X trimestre YYYY", where X is the trimester number (1, 2, 3, or 4), and YYYY is the year.

    Returns:
        str or None: The corresponding quarter string in the format "YYYY-TX", where X is the trimester number, and YYYY is the year. Returns None if the input string is not in the correct format.
    """
    regex = re.match(r'([1-4])[a-z]* trimestre ([0-9]{4})', trimestre)
    if regex:
        quarter = regex.group(1)
        year = regex.group(2)
        return f'{year}-T{quarter}'
    else:
        return None

def swap_quarter_and_year(trimestre):
    """
    Swap the year and quarter in a trimester string.

    Parameters:
        trimestre (str): The trimester string in the format "YYYY-TX".

    Returns:
        str: The corresponding trimester string in the format "X trimestre YYYY", where X is the quarter number (1, 2, 3, or 4), and YYYY is the year.
    """
    regex = re.match(r'T([0-9]{1})_([0-9]{4})', trimestre)
    if regex:
        year = regex.group(2)
        quarter = regex.group(1)
        return f'{year}-T{quarter}'
    else:
        return None

################################
# Beginning of the script
################################

base_path = '/home/eric/Documents/Websites/eric-roca.github.io/'
labour_url = 'https://www.insee.fr/fr/statistiques/fichier/2012804/sl_etc_2024T1.xls'
url_crises = 'https://www.afse.fr/global/gene/link.php?doc_id=426&fg=1'
url_pib = 'https://www.insee.fr/fr/statistiques/fichier/2830547/econ-gen-pib-composante-trim.xlsx'

crises_archive = '/home/eric/Documents/Websites/eric-roca.github.io/content/courses/Macroeconomics/Macro_2/Archive/2021162908_dates-afse.xlsx'
labour_archive = '/home/eric/Documents/Websites/eric-roca.github.io/content/courses/Macroeconomics/Macro_2/Archive/sl_etc_2024T1.xlsx'

# Plot the French crises
# Data for dating the crises from https://www.afse.fr/fr/cycles-eco/dates-500216
# Archived on 2024-07-23


# Try to download the data, if it fails, use the archived version
try:
    response = requests.get(url_crises)
    crises = pd.read_excel(response.content, sheet_name='Dates')
except:
    crises_archive = pd.read_excel(archive, sheet_name='Dates')

crises.rename(columns = {'Unnamed: 0' : 'Quarter',
                        'Dates ' : 'Recessions'}, inplace = True)

pib = pd.read_excel(url_pib, skiprows=3)
pib.rename(columns = {'Trimestre' : 'Quarter', 
                        'Produit intérieur brut (PIB)' : 'PIB',
                        'Dépense de consommation des ménages' : 'Consumption',
                        'Formation brute de capital fixe' : 'K'}, inplace = True)

pib.drop(pib.loc[pib['PIB'].isna()].index, inplace=True)
pib = pib[['Quarter', 'PIB', 'Consumption', 'K']]
pib['Date'] = pib['Quarter'].apply(year_trimester_to_date)

pib = pib.merge(crises, on='Quarter', how='left')

outpath = os.path.join(base_path, 'static/img/macro_2', 'french_recessions_pib.webp')
plot_pib_with_recessions(pib, variable='PIB', variable_label='Changement du PIB (%)', display_mean=True, color='purple', outpath=outpath)

outpath = os.path.join(base_path, 'static/img/macro_2', 'french_recessions_consumption.webp')
plot_pib_with_recessions(pib, variable='Consumption', variable_label='Changement de la consommation (%)', display_mean=True, color='purple', outpath=outpath)

outpath = os.path.join(base_path, 'static/img/macro_2', 'french_recessions_k.webp')
plot_pib_with_recessions(pib, variable='K', variable_label='Changement de la formation brute de capital fixe (%)', display_mean=True, color='purple', outpath=outpath)

try:
    response = requests.get(labour_url)
    labour = pd.read_excel(labour_url, sheet_name='Région', skiprows=3)
except:
    labour = pd.read_excel(labour_archive, sheet_name='Région', skiprows=3)

labour = labour.loc[labour['Libellé'] == 'FRANCE METROPOLITAINE']
labour.drop(columns = ['Code'], inplace = True)
# Pivot from wide to long
labour = labour.melt(id_vars='Libellé', var_name='Quarter', value_name='Unemployment')

# Quarters appear as T*-YYYY, swap to have YYYY-T*
labour['Quarter'] = labour['Quarter'].apply(swap_quarter_and_year)

pib = pib.merge(labour[['Quarter', 'Unemployment']], on='Quarter', how='left')

outpath = os.path.join(base_path, 'static/img/macro_2', 'french_unemployment.webp')
plot_pib_with_recessions(pib, variable='Unemployment', variable_label='Taux de chômmage (%)', display_mean=True, color='purple', outpath=outpath)