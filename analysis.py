#!/usr/bin/env python3 
#coding: utf8

## Week 3 Project for Stats 140SL.
## Analyze effects of state tax on economic recovery

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from matplotlib.dates import DateFormatter
from scipy.stats import ttest_ind

warnings.filterwarnings('ignore')

def load_data(make_csv = False):
    """
    args: none
    returns: state employment and tax data 
    """

    ## Load tax data and get combinedRate
    tax = pd.read_csv('data/state_tax.csv', delimiter=',')

    ## Load all employment data merged with geo_state data
    employment = pd.read_csv('data/state_employment.csv', delimiter=',')
    geo_state = pd.read_csv('data/geo_state.csv', delimiter=',')
    merged_data = employment.merge(geo_state, on='statefips')

    ## Now merge with tax data
    clean_data = merged_data.merge(tax, on='statename')

    # If we want to output the CSV of the data
    if(make_csv):
        compression_opts = dict(method='zip', archive_name='out.csv')  
        merged_data.to_csv('out.zip', index=False, compression=compression_opts) 

    return (clean_data)

def plot_main(df, isLow=False, isMed=False, isHigh=False, is40=False, is60=False, is65=False, is70=False):
    """
    output plot of states with top and bottom 5 tax rates and their employment recovery.
    arg df: clean data obtained from load_data
    returns: nothing
    """

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
    plt.rc('font', **font)

    ## Get top 5 tax rate states
    top5 = df[df['combinedRate'] >= df['combinedRate'].quantile(.90)]
    bot5 = df[df['combinedRate'] <= df['combinedRate'].quantile(.10)]

    ## Convert three separate date attribute columns into one.
    dates = pd.to_datetime(top5[['year', 'month', 'day']])

    # update top and bottom 5 with dates
    top5 = pd.DataFrame(top5)
    top5['dates'] = dates
    bot5 = pd.DataFrame(bot5)
    bot5['dates'] = dates

    # Create figure and plot space
    fig, ax = plt.subplots(figsize=(12, 12))
      
    if(isLow):
        # For low income
        ax.bar(dates, top5['emp_combined_inclow'], color='red', label="Highest 5 Tax")
        ax.bar(dates, bot5['emp_combined_inclow'], color='blue', alpha=0.3, label="Lowest 5 Tax")
        ax.set(xlabel="Date", ylabel="Employment level for Low Income Workers (under $27,000)",
               title="COVID Impact on Employment of Top and Bottom 5 Taxed States")
        print(top5['emp_combined_inclow'].isnull().sum())
    elif(isMed):
        # For middle income
        ax.bar(dates, top5['emp_combined_incmiddle'], color='red', label="Highest 5 Tax")
        ax.bar(dates, bot5['emp_combined_incmiddle'], color='blue', alpha=0.3, label="Lowest 5 Tax")
        ax.set(xlabel="Date", ylabel="Employment level for Mid Income Workers ($27k to $60k)",
               title="COVID Impact on Employment of Top and Bottom 5 Taxed States")
    elif(isHigh):
        # For high income
        top5 = top5[top5['emp_combined_inchigh'] != '.']
        top_y = pd.to_numeric(top5['emp_combined_inchigh'])
        bot5 = bot5[bot5['emp_combined_inchigh'] != '.']
        bot_y = pd.to_numeric(bot5['emp_combined_inchigh'])
        ax.bar(top5['dates'], top_y, color='red', label="Highest 5 Tax")
        ax.bar(dates, bot_y, color='blue', alpha=0.3, label="Lowest 5 Tax")
        ax.set(xlabel="Date", ylabel="Employment level for Top Income Workers (over $60,000)",
               title="COVID Impact on Employment of Top and Bottom 5 Taxed States")
    elif(is40):
        # For workers in trade, transportation, and utilities
        top5 = top5[top5['emp_combined_ss40'] != '.']
        top_y = pd.to_numeric(top5['emp_combined_ss40'])
        bot5 = bot5[bot5['emp_combined_ss40'] != '.']
        bot_y = pd.to_numeric(bot5['emp_combined_ss40'])
        ax.bar(top5['dates'], top_y, color='red', label="Highest 5 Tax")
        ax.bar(dates, bot_y, color='blue', alpha=0.3, label="Lowest 5 Tax")
        ax.set(xlabel="Date", ylabel="Employment level for Workers in trade, transportation and utilities",
               title="COVID Impact on Employment of Top and Bottom 5 Taxed States")
    elif(is60):
        # For workers in professional and business services
        top5 = top5[top5['emp_combined_ss60'] != '.']
        top_y = pd.to_numeric(top5['emp_combined_ss60'])
        bot5 = bot5[bot5['emp_combined_ss60'] != '.']
        bot_y = pd.to_numeric(bot5['emp_combined_ss60'])
        ax.bar(top5['dates'], top_y, color='red', label="Highest 5 Tax")
        ax.bar(dates, bot_y, color='blue', alpha=0.3, label="Lowest 5 Tax")
        ax.set(xlabel="Date", ylabel="Employment level for Workers in professional and business services",
               title="COVID Impact on Employment of Top and Bottom 5 Taxed States")
    elif(is65):
        # For workers in education and health services
        top5 = top5[top5['emp_combined_ss65'] != '.']
        top_y = pd.to_numeric(top5['emp_combined_ss65'])
        bot5 = bot5[bot5['emp_combined_ss65'] != '.']
        bot_y = pd.to_numeric(bot5['emp_combined_ss65'])
        ax.bar(top5['dates'], top_y, color='red', label="Highest 5 Tax")
        ax.bar(dates, bot_y, color='blue', alpha=0.3, label="Lowest 5 Tax")
        ax.set(xlabel="Date", ylabel="Employment level for Workers in  education and health services",
               title="COVID Impact on Employment of Top and Bottom 5 Taxed States")
    elif(is70):
        # For workers in education and health services
        top5 = top5[top5['emp_combined_ss70'] != '.']
        top_y = pd.to_numeric(top5['emp_combined_ss70'])
        bot5 = bot5[bot5['emp_combined_ss70'] != '.']
        bot_y = pd.to_numeric(bot5['emp_combined_ss70'])
        ax.bar(top5['dates'], top_y, color='red', label="Highest 5 Tax")
        ax.bar(dates, bot_y, color='blue', alpha=0.3, label="Lowest 5 Tax")
        ax.set(xlabel="Date", ylabel="Employment level for Workers in leisure and hospitality",
               title="COVID Impact on Employment of Top and Bottom 5 Taxed States")
    else:
        # For all workers overall
        ax.bar(dates, top5['emp_combined'], color='red', label="Highest 5 Tax")
        ax.bar(dates, bot5['emp_combined'], color='blue', alpha=0.3, label="Lowest 5 Tax")
        ax.set(xlabel="Date", ylabel="Employment level for all workers",
        title="COVID Impact on Employment of Top and Bottom 5 Taxed States")
    
    ax.legend()

    # Define the date format
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    
    plt.grid()
    plt.show()

def t_test(df):
    """
    perform t-test and see if the rate can be explained randomly or not.
    methodology: 1. take minimum employment and compare to current day.
                 2. compute the difference between the two and set as 
                    our recovery value
                 3. compute t-test between the top 20% and bottom 20%
                    of this recovery value. 
    for now, just doing this with emp_combined.
    arg df: clean data obtained from load_data
    returns: dictionary of p-values for each of the t-tests performed
    """

    top10 = df[df['combinedRate'] >= df['combinedRate'].quantile(.80)]
    bot10 = df[df['combinedRate'] <= df['combinedRate'].quantile(.20)]

    emp_labels = ['emp_combined', 'emp_combined_inclow', 'emp_combined_incmiddle',
                  'emp_combined_inchigh', 'emp_combined_ss40', 'emp_combined_ss60',
                  'emp_combined_ss65', 'emp_combined_ss70']
    p_values = []

    for label in emp_labels:
        top10_clean = top10[top10[label] != '.']
        top10_emp = top10_clean.groupby(['statename'], sort=False)[label].min()
        top10_cur = top10_clean[np.logical_and(top10['month'] == 9, top10['day'] == 3)][['statename', label]]
        x = list(map(float,top10_emp.tolist()))
        y = list(map(float,top10_cur[label].tolist()))
        a =  [y_i - x_i for y_i, x_i in zip(y, x)]

        bot10_clean = bot10[bot10[label] != '.']
        bot10_emp = bot10_clean.groupby(['statename'], sort=False)[label].min()
        bot10_cur = bot10_clean[np.logical_and(bot10['month'] == 9, bot10['day'] == 3)][['statename', label]]
        x = list(map(float,bot10_emp.tolist()))
        y = list(map(float,bot10_cur[label].tolist()))
        b =  [y_i - x_i for y_i, x_i in zip(y, x)]

        t, p_val = ttest_ind(a, b, equal_var=False)
        p_values.append(p_val)

    return (dict(zip(emp_labels, p_values)) )


if __name__ == "__main__":
    clean_data = load_data()
    #plot_main(clean_data)

    # Null hypothesis that the mean recovery rate between top 
    # and bottom 10 states based on tax rate is equal.
    # P value is less than significance level of 0.05
    # so we rejected our null hypothesis. 
    # We conclude the means are different, thus tax rate
    # has an effect on recovery rate. 
    p = t_test(clean_data)
    print(p)