# Labormatics-Covid-Conference
Labormatics Code and data for upcoming Covid Conference
Performed in Python. Analyzes the impact of Covid-19 on American economy by analyzing the stock and revenue data. 
Daily stock data from 8-10-20 to 5 years in the past was scraped from Yahoo Finance (https://finance.yahoo.com/) for the top 25 companies by stock per economic sector.
Quarterly revenue data from July to 5 years in the past was scraped from Macro Trends (https://www.macrotrends.net) for the top 5 companies by stock per economic sector.
Covid-19 cases were pulled from https://ourworldindata.org/coronavirus-source-data.
Web scraping from Yahoo Finance heavily inspired by https://medium.com/c%C3%B3digo-ecuador/how-to-scrape-yahoo-stock-price-history-with-python-b3612a64bdc6

Each economic sector was pulled in as a list of their top 25 companies by stock.
Yahoo finance data was then scraped into each company.
They are then graphed against covid 19 cases and against time.
They are also graphed in the same manner with truncated version of the data from November 2019 to present to see a "zoomed in" version of the dip in stocks from covid.
A trend line is constructed with the data preceding the outbreak date of Covid-19 to analyze the general direction of stock value data in the absence of Covid.

Checks are made to see how many of the sampled companies experience a 5 year low record.

