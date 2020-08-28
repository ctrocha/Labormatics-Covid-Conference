#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import statistics
#!pip install -U scikit-learn
import sklearn
import inspect
from sklearn import linear_model
from datetime import datetime, timedelta 
import time
from pandas import Timestamp
import requests, pandas, lxml
from lxml import html
def format_date(date_datetime):
     date_timetuple = date_datetime.timetuple()
     date_mktime = time.mktime(date_timetuple)
     date_int = int(date_mktime)
     date_str = str(date_int)
     return date_str
def subdomain(symbol, start, end, filter='history'):
     subdoma="/quote/{0}/history?period1={1}&period2={2}&interval=1d&filter={3}&frequency=1d"
     subdomain = subdoma.format(symbol, start, end, filter)
     return subdomain
 
def header_function(subdomain):
     hdrs =  {"authority": "finance.yahoo.com",
              "method": "GET",
              "path": subdomain,
              "scheme": "https",
              "accept": "text/html",
              "accept-encoding": "gzip, deflate, br",
              "accept-language": "en-US,en;q=0.9",
              "cache-control": "no-cache",
              "cookie": "Cookie:identifier",
              "dnt": "1",
              "pragma": "no-cache",
              "sec-fetch-mode": "navigate",
              "sec-fetch-site": "same-origin",
              "sec-fetch-user": "?1",
              "upgrade-insecure-requests": "1",
              "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64)"}
     
     return hdrs
def scrape_page(url, header):
     page = requests.get(url, headers=header)
     element_html = html.fromstring(page.content)
     table = element_html.xpath('//table')
     table_tree = lxml.etree.tostring(table[0], method='xml')
     panda = pandas.read_html(table_tree)
     return panda
if __name__ == '__main__':
     symbol = 'BB'
     
     dt_start = datetime.today() - timedelta(days=365)
     dt_end = datetime.today()
    
     start = format_date(dt_start)
     end = format_date(dt_end)
     
     sub = subdomain(symbol, start, end)
     header = header_function(sub)
     
     base_url = 'https://finance.yahoo.com'
     url = base_url + sub
     price_history = scrape_page(url, header)


# In[2]:


import re
from io import StringIO
from datetime import datetime, timedelta

import requests
import pandas as pd


class YahooFinanceHistory:
    timeout = 2
    crumb_link = 'https://finance.yahoo.com/quote/{0}/history?p={0}'
    crumble_regex = r'CrumbStore":{"crumb":"(.*?)"}'
    quote_link = 'https://query1.finance.yahoo.com/v7/finance/download/{quote}?period1={dfrom}&period2={dto}&interval=1d&events=history&crumb={crumb}'

    def __init__(self, symbol, days_back=7):
        self.symbol = symbol
        self.session = requests.Session()
        self.dt = timedelta(days=days_back)

    def get_crumb(self):
        response = self.session.get(self.crumb_link.format(self.symbol), timeout=self.timeout)
        response.raise_for_status()
        match = re.search(self.crumble_regex, response.text)
        if not match:
            raise ValueError('Could not get crumb from Yahoo Finance')
        else:
            self.crumb = match.group(1)

    def get_quote(self):
        if not hasattr(self, 'crumb') or len(self.session.cookies) == 0:
            self.get_crumb()
        now = datetime.utcnow()
        dateto = int(now.timestamp())
        datefrom = int((now - self.dt).timestamp())
        url = self.quote_link.format(quote=self.symbol, dfrom=datefrom, dto=dateto, crumb=self.crumb)
        response = self.session.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), parse_dates=['Date'])


# In[ ]:





# In[3]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

Basic_Materials_Sector = ['BHP','LIN','BBL','RIO','CTA-PB','CTA-PA','APD','ECL','VALE','SHW','NEM','GOLD','DD','SCCO','DOW','CRH','FNV','PPG','WPM','LYB','CTVA',
'FCX','NTR','VMC','AEM']

Communication_Services_Sector =['CABO','GOOG','GOOGL','CHTR','NFLX','NTES','SPOT','TWLO','ZM','FB','MSGS','ROKU','TTWO','EA','LBRDA','IAC','BIDU','DIS','SE','TMUS','MTCH','FVRR','CCOI','YY']

Consumer_Cyclical_Sector = ['NVR','AMZN','BKNG','TSLA','CMG','AZO','MELI','ORLY','DPZ','BH-A','LULU','POOL','RH','HD','BABA','W','ULTA','DECK','MTN','MCD','BURL','CVCO','RACE','LAD','MED']

Consumer_Defensive_Sector = ['SAM','GHC','COST','CLX','COKE','EL','MKC-V','MKC','DG','HELE','STZ-B','STZ','STRA','CASY','LANC','EDU','KMB','DEO','HSY','PEP','WMT','BYND','JJSF','PG','TGT']

Financial_Services_Sector = ['BRK-A','FNMFO','MCHB','AFIPA','FINN','FMBL','SBBG','ALPIB','SBNC','BHRB','BAC-PL','WFC-PL','VRTB','MKL','PGPHF','QUCT','WTM','FMIA','USB-PA','FMCB','CBCY','CBCYB','BLK','BKUT','Y']

Real_Estate_Sector = ['TPRP','LAACZ','BVERS','MNPP','CSGP','EQIX','GDVTZ','SBAC','ALX','AMT','AZLCZ','ESS','PSA','ARE','CCI','QTS-PB','AVB','DLR','LEGIF','SUI','PSB','EGP','COR','MAA','FSV']

Healthcare_Sector = ['DHR-PA','DHR-PB','BMYMP','SAUHF','MTD','ERFSF','ISRG','ATRI','DWRKF','REGN','LZAGF','BIO-B','CHE','BIO','HUM','DXCM','TMO','TFX','IDXX','RHHBF','ILMN','RHHVF','GNMSF','UNH','SDMHF']

Utilities_Sector = ['NEE','BANGN','WELPM','PPWLM','DOGEF','AWK','AILNP','SRE','CTPPO','DTE','CMS-PB','SRE-PB','SRE-PA','IPWLK','NMK-PC','ALPVN','ATO','APRCP','UEPEO','ETR','PNMXO','DCUE','AILLM','APRDN','UEPEP']

Energy_Sector = ['TPL','AMEN','MITSY','CCRK','TRKX','NOVKY','RSRV','PDER','CEO','PXD','CVX','TTYP','PNRG','LUKOY','REX','PSX','VOPKY','VOPKF','NBR','VLO','LNG','HES','CXO','EOG','TRP']

Industrials_Sector = ['SEB','CIBY','AMKBF','AMKAF','BKDKP','DUAVF','GBERF','SMECF','TDG','ROP','FCELB','LMT','GWW','UHAL','NOC','CTAS','CP','NSYC','LII','SHLAF','SHLRF','WSO','WSO-B','ROK','CMI']

Technology_Sector =  ['ADYYF','AVGOP','CNSWF','SHOP','FTV-PA','TTD','AAPL','NVDA','ADBE','NOW','FICO','KYCCF','LRCX','ASML','ASMLF','TYL','TDY','AVGO','ANSS','EPAM','INTU','PAYC','COUP','RNG','ZBRA']


# In[ ]:





# ## YahooFinanceHistory code is old copy. Use
# def clean_data(stock_data, col):
#     weekdays = pd.date_range(start=START_DATE, end=END_DATE)
#     clean_data = stock_data[col].reindex(weekdays)
#     return clean_data.fillna(method='ffill')
# 
# def get_data(ticker):
#     try:
#         stock_data = data.DataReader(ticker,
#                                     'yahoo',
#                                     START_DATE,
#                                     END_DATE)
#         adj_close = clean_data(stock_data, 'Open')
#         return adj_close 
#     
#     except RemoteDataError:
#         print('No data for {t}'.format(t=ticker))
#         
#         
#         
# for sectInx, sect in enumerate(Sectors,0):
#     plt.subplots(figsize=(12,8))
#     plt.title(Sectors_names[sectInx])
#     for company in sect:
#         plt.plot(get_data(company))
#     plt.show()
# 
# 
# import yfinance as yf #yahoo finance specific package, formerly modification now stand alone.
# 
# sbux = yf.Ticker("SBUX")
# tlry = yf.Ticker("TLRY")
# 
# print(sbux.info['sector'])
# print(tlry.info['sector'])
# 
# import yfinance as yf
# 
# sbux = yf.Ticker("SBUX")
# tlry = yf.Ticker("TLRY")
# 
# print(sbux.info['sector'])
# print(tlry.info['sector'])

# In[4]:


def Sector_Appender(Sector_list,Sector_name):
    
    table = YahooFinanceHistory(Sector_list[0], days_back=(365*5)).get_quote()
    Sector_list[0] = YahooFinanceHistory(Sector_list[0], days_back=(365*5)).get_quote()
    giant_frame = Sector_list[0]
    giant_frame_sum = Sector_list[0]
    giant_frame_sum_cols = giant_frame_sum[['Open','High','Low','Close',]]
    for index in range(1,len(Sector_list)): 
        data = YahooFinanceHistory(Sector_list[index], days_back=(365*5)).get_quote()
        Sector_list[index] = data
        giant_frame = giant_frame.append(data)
        giant_frame_sum_cols = giant_frame_sum[['Open','High','Low','Close',]]+data[['Open','High','Low','Close',]]
        table = table.append(data)
    Sector_name_csv = Sector_name+"_csv"
    giant_frame.to_csv(Sector_name_csv)
    giant_frame_sum[['Open','High','Low','Close',]] = giant_frame_sum_cols[['Open','High','Low','Close',]]
    giant_frame_average_name = Sector_name+"_average.csv"
    #giant_frame_average = giant_frame_sum
    giant_frame_sum[['Open','High','Low','Close']] =  giant_frame_sum[['Open','High','Low','Close']].div(25)
    giant_frame_sum.to_csv(giant_frame_average_name)
    giant_frame_sum
    return Sector_list




























def excel_Appender(Sector_list,name):
    table_list = Sector_Appender(Sector_list)
    giant_frame = table_list[0]
    sheet[0].to_csv(name)
    for index in range(1,len(Sector_list)):
        sheet[index].to_csv(name.csv, mode = 'a', header=False)

def stock_analyzer_group(Sector,Sector_name):
    #graph = plt.plot(Sector[0]['Date'],Sector[0]['Open'])
    for company in range(0,len(Sector)):
        #print(Sector[company])
        #stock_analyzer(Sector[company])
        #plt.figure(figsize=(3,4))
        plt.plot(Sector[company]['Date'],Sector[company]['Open'])
        
        #return graph
   # plt.legend(['BHP','LIN','BBL','RIO','CTA-PB','CTA-PA','APD','ECL','VALE','SHW','NEM','GOLD','DD','SCCO','DOW','CRH','FNV','PPG','WPM','LYB','CTVA','FCX','NTR','VMC','AEM'])
    plt.title(Sector_name)
    return plt


# In[ ]:





# In[245]:


Basic_Materials_table_list = Sector_Appender(Basic_Materials_Sector,'Basic_Materials_Sector')


# In[246]:


Communication_Services_table_list = Sector_Appender(Communication_Services_Sector,'Communication_Services_Sector')


# In[247]:


Consumer_Cyclical_table_list = Sector_Appender(Consumer_Cyclical_Sector,'Consumer_Cyclical_Sector')


# In[248]:


Financial_Services_Sector_table_list = Sector_Appender(Financial_Services_Sector,'Financial_Services_Sector')


# In[249]:


Real_Estate_Sector_table_list = Sector_Appender(Real_Estate_Sector,'Real_Estate_Sector')


# In[253]:


Consumer_Defensive_table_list = Sector_Appender(Consumer_Defensive_Sector,'Consumer_Defensive_Sector')


# In[254]:


Healthcare_Sector_table_list = Sector_Appender(Healthcare_Sector,'Healthcare_Sector')


# In[255]:


Utilities_Sector_table_list = Sector_Appender(Utilities_Sector,'Utilities_Sector')


# In[261]:


Energy_Sector_table_list =Sector_Appender(Energy_Sector,'Energy_Sector')


# In[262]:


Industrials_Sector_table_list =Sector_Appender(Industrials_Sector,'Industrials_Sector')


# In[263]:


Technology_Sector_table_list =Sector_Appender(Technology_Sector,'Technology_Sector')


# In[46]:


from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


# In[ ]:





# In[ ]:





# In[55]:





a = stock_analyzer_group(Basic_Materials_table_list,"Basic_Materials")
a.show()
b = stock_analyzer_group(Communication_Services_table_list,"Communication_Services")
b.show()
c = stock_analyzer_group(Consumer_Cyclical_table_list,"Consumer_Cyclical")
c.show()
d = stock_analyzer_group(Financial_Services_Sector_table_list,"Finance")
d.show()
e= stock_analyzer_group(Consumer_Defensive_table_list,"Consumer_Defensive_table_list")
e.show()
f= stock_analyzer_group(Real_Estate_Sector_table_list,"Real_Estate_Sector_table_list")
f.show()
g= stock_analyzer_group(Healthcare_Sector_table_list,"Healthcare_Sector_table_list")
g.show()
h= stock_analyzer_group(Utilities_Sector_table_list,"Utilities_Sector_table_list")
h.show()
i = stock_analyzer_group(Energy_Sector_table_list,"Energy_Sector_table_list")
i.show()


# In[104]:


def stock_analyzer_group_Trun(Sector,Sector_name):
    #graph = plt.plot(Sector[0]['Date'],Sector[0]['Open'])
    for company in range(0,len(Sector)):

        November_End_index = (list(Sector[company]['Date']).index(Timestamp('2019-11-19 00:00:00')))
        #print(November_End_index)
        Sector[company] = Sector[company][November_End_index:1300]
        #print(Sector[company]['Date'])
        plt.plot(Sector[company]['Date'],Sector[company]['Open'])
        plt.title(Sector_name)
        #print(list(Sector[company]['Date']))
    fig = plt.figure()
    fig.set_figheight(12)
    fig.set_figwidth(12)
    return fig


a = stock_analyzer_group_Trun(Basic_Materials_table_list,"Basic_Materials")
a.show()
#b = stock_analyzer_group_Trun(Communication_Services_table_list,"Communication_Services")
#b.show()
c = stock_analyzer_group_Trun(Consumer_Cyclical_table_list,"Consumer_Cyclical")
c.show()
d = stock_analyzer_group_Trun(Financial_Services_Sector_table_list,"Finance")
d.show()
e= stock_analyzer_group_Trun(Consumer_Defensive_Sector,"Consumer_Defensive_table_list")
e.show()
f= stock_analyzer_group_Trun(Real_Estate_Sector_table_list,"Real_Estate_Sector_table_list")
f.show()
#g= stock_analyzer_group_Trun(Healthcare_Sector_table_list,"Healthcare_Sector_table_list")
#g.show()
h= stock_analyzer_group_Trun(Utilities_Sector_table_list,"Utilities_Sector_table_list")
h.show()
i = stock_analyzer_group_Trun(Energy_Sector_table_list,"Energy_Sector_table_list")
i.show()


# In[107]:


Consumer_Cyclical_average = pd.read_csv('Consumer_Cyclical_Sectoraverage')
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Consumer_Cyclical_average.plot(x ='Date', y='Open', kind = 'scatter')
plt.show()


# In[273]:


plt.rcParams['figure.figsize'] = [20, 5] #Larger output graph
Basic_Materials_table_list
Basic_Materials_Sector_average = pd.read_csv('Basic_Materials_Sector_average')  #Read from CSV
Basic_Materials_Sectoraverage.plot(x ='Date', y='Open', kind = 'line') #Construct graph via Matplotlib
plt.title('Basic_Materials_Sector_average')

plt.savefig('Basic_Materials_Sector_average.png', bbox_inches='tight')


# In[272]:


plt.rcParams['figure.figsize'] = [20, 5]
Basic_Materials_table_list
Communication_Services_Sector_average = pd.read_csv('Communication_Services_Sector_average')
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Communication_Services_Sector_average.plot(x ='Date', y='Open', kind = 'line')
plt.title('Communication_Services_Sector_average')

plt.savefig('Communication_Services_Sector_average.png', bbox_inches='tight')


# In[274]:


plt.rcParams['figure.figsize'] = [20, 5]
Basic_Materials_table_list
Consumer_Cyclical_Sector_average = pd.read_csv('Consumer_Cyclical_Sector_average')
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Consumer_Cyclical_Sector_average.plot(x ='Date', y='Open', kind = 'line')
plt.title('Consumer_Cyclical_Sector_average')

plt.savefig('Consumer_Cyclical_Sector_average.png', bbox_inches='tight')


# In[270]:


plt.rcParams['figure.figsize'] = [20, 5]
Basic_Materials_table_list
Consumer_Defensive_Sector_average = pd.read_csv('Consumer_Defensive_Sector_average')
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Consumer_Defensive_Sector_average.plot(x ='Date', y='Open', kind = 'line')
plt.title('Consumer_Defensive_Sector_average')

plt.savefig('Consumer_Defensive_Sector_average.png', bbox_inches='tight')


# In[269]:


plt.rcParams['figure.figsize'] = [20, 5]
Basic_Materials_table_list
Financial_Services_Sector_average = pd.read_csv('Financial_Services_Sector_average')
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Financial_Services_Sector_average.plot(x ='Date', y='Open', kind = 'line')
plt.title('Financial_Services_Sector_average')

plt.savefig('Financial_Services_Sector_average.png', bbox_inches='tight')


# In[176]:


plt.rcParams['figure.figsize'] = [20, 5]
Basic_Materials_table_list
Real_Estate_Sector_average = pd.read_csv('Real_Estate_Sector_average')
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Real_Estate_Sector_average.plot(x ='Date', y='Open', kind = 'line')
plt.title('Basic_Materials_Sector_average')


# In[177]:


data:image/png;base64,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plt.rcParams['figure.figsize'] = [20, 5]
Basic_Materials_table_list
Healthcare_Sector_average = pd.read_csv('Healthcare_Sector_average')
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Healthcare_Sector_average.plot(x ='Date', y='Open', kind = 'line')
plt.title('Basic_Materials_Sector_average')


# In[178]:


plt.rcParams['figure.figsize'] = [20, 5]
Basic_Materials_table_list
Utilities_Sector_average = pd.read_csv('Utilities_Sector_average')
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Utilities_Sector_average.plot(x ='Date', y='Open', kind = 'line')
plt.title('Basic_Materials_Sector_average')


# In[199]:


plt.rcParams['figure.figsize'] = [20, 5]
Basic_Materials_table_list
Energy_Sector_average = pd.read_csv('Energy_Sector_average')
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Energy_Sector_average.plot(x ='Date', y='Open', kind = 'line')
plt.title('Basic_Materials_Sector_average')


# In[198]:


plt.rcParams['figure.figsize'] = [20, 5]
Basic_Materials_table_list
Industrials_Sector_average = pd.read_csv('Industrials_Sector_average')
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Industrials_Sector_average.plot(x ='Date', y='Open', kind = 'line')
plt.title('Basic_Materials_Sector_average')


# In[204]:


plt.rcParams['figure.figsize'] = [20, 5]
Basic_Materials_table_list
Technology_Sector_average = pd.read_csv('Technology_Sector_average')
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Technology_Sector_average.plot(x ='Date', y='Open', kind = 'line')
plt.title('Basic_Materials_Sector_average')


# In[211]:


plt.rcParams['figure.figsize'] = [20, 5]
Basic_Materials_Sector_average_trun = Basic_Materials_Sector_average[846:1259]
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Basic_Materials_Sector_average_trun.plot(x ='Date', y='Open', kind = 'line')
plt.title('Basic_Materials_Sector_average_trun')


# In[212]:


plt.rcParams['figure.figsize'] = [20, 5]
Communication_Services_Sector_average_trun = Communication_Services_Sector_average[846:1259]
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Communication_Services_Sector_average_trun.plot(x ='Date', y='Open', kind = 'line')
plt.title('Communication_Services_Sector_average_trun')


# In[213]:


plt.rcParams['figure.figsize'] = [20, 5]
Consumer_Cyclical_Sector_average_trun = Consumer_Cyclical_Sector_average[846:1259]
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Consumer_Cyclical_Sector_average_trun.plot(x ='Date', y='Open', kind = 'line')
plt.title('Consumer_Cyclical_Sector_average_trun')


# In[214]:


plt.rcParams['figure.figsize'] = [20, 5]
Consumer_Defensive_Sector_average_trun = Consumer_Defensive_Sector_average[846:1259]
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Consumer_Defensive_Sector_average_trun.plot(x ='Date', y='Open', kind = 'line')
plt.title('Consumer_Defensive_Sector_average_trun')


# In[215]:


plt.rcParams['figure.figsize'] = [20, 5]
Financial_Services_Sector_average_trun = Financial_Services_Sector_average[846:1259]
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Financial_Services_Sector_average_trun.plot(x ='Date', y='Open', kind = 'line')
plt.title('Financial_Services_Sector_average_trun')


# In[219]:


plt.rcParams['figure.figsize'] = [20, 5]
Real_Estate_Sector_average_trun = Real_Estate_Sector_average[846:1259]
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Real_Estate_Sector_average_trun.plot(x ='Date', y='Open', kind = 'line')
plt.title('Real_Estate_Sector_average_trun')


# In[ ]:


plt.rcParams['figure.figsize'] = [20, 5]
Basic_Materials_Sector_average_trun = Basic_Materials_Sector_average[846:1259]
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Basic_Materials_Sector_average_trun.plot(x ='Date', y='Open', kind = 'line')
plt.title('Basic_Materials_Sector_average_trun')


# In[ ]:


plt.rcParams['figure.figsize'] = [20, 5]
Basic_Materials_Sector_average_trun = Basic_Materials_Sector_average[846:1259]
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Basic_Materials_Sector_average_trun.plot(x ='Date', y='Open', kind = 'line')
plt.title('Basic_Materials_Sector_average_trun')


# In[ ]:


plt.rcParams['figure.figsize'] = [20, 5]
Basic_Materials_Sector_average_trun = Basic_Materials_Sector_average[846:1259]
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Basic_Materials_Sector_average_trun.plot(x ='Date', y='Open', kind = 'line')
plt.title('Basic_Materials_Sector_average_trun')


# In[ ]:


plt.rcParams['figure.figsize'] = [20, 5]
Basic_Materials_Sector_average_trun = Basic_Materials_Sector_average[846:1259]
#caverage = stock_analyzer_group_Trun(Consumer_Cyclical_average,"Consumer_Cyclical_Sectoraverage")
Basic_Materials_Sector_average_trun.plot(x ='Date', y='Open', kind = 'line')
plt.title('Basic_Materials_Sector_average_trun')


# In[ ]:


#RCL.Date[RCL.Date == "2020-01-02"].index.tolist()
#np.std(list(RCL["High"]))
#RCL_trunc = RCL[0:1108]
#RCL_trunc

def stock_analyzer(Company):
    
    USA_COVID_date = Company.Date[Company.Date == "2020-01-02"].index.tolist() #Returns index of the "2020-01-02"
    Company_prior = Company[0:USA_COVID_date[0]]
    Company_post = Company[USA_COVID_date[0]:1257]
    
    Company_prior_Open_min = min(list(Company_prior["Open"]))
    Company_post_Open_min = min(list(Company_post["Open"]))
#    years = [1,2,3,4,5]0 
    if Company_post_Open_min < Company_prior_Open_min:
        #company_name = inspect.getfullargspec(stock_analyzer)[0]
        print("Experienced 5 year low record")
  #      print(str(Company))
    else: 
        print(str(Company))
    return("swaggggggggggg")    
    #plt.plot(Company['Date'],Company['Open'])
    


# In[ ]:


table4 = table1.append(table2)
table4 = table4.append(table3)
#table5 = 
#plt.plot(table4['Date'],table4['Open'])
#sns.lineplot(x="Date", y="Open", data=table4)
total_cases = pd.read_csv('total_cases.csv')
new_cases = pd.read_csv('new_cases.csv')
total_casesymd = pd.read_csv('total_casesymd.csv')
total_cases = total_cases[2:225]
total_cases = total_cases[["Date","World","United States"]]
total_casesymd = total_casesymd[["Date","World","United States"]]
total_cases 
USA_COVID_date = table1.Date[table1.Date == "2020-01-02"].index.tolist()
table1_covid = table1[0:11]

total_cases

print(type(table1["Date"][0]))

total_cases

total_casesymd['Date'] = pd.to_datetime(total_casesymd['Date'])
new_cases['Date'] = pd.to_datetime(new_cases['Date'])
#table_concat = pd.merge(table1, total_casesymd, how = 'inner', left_index = True, right_index=True)
table_concat = pd.merge(table1, total_casesymd, how = 'left', on='Date')
#df = pd.merge(left_frame, right_frame, how='right', on='key')
table_concat
table_new_cases_concat = pd.merge(table1, new_cases, how = 'left', on='Date')
table_new_cases_concat = table_new_cases_concat.fillna(0)
table_new_cases_concat
swag  = plt.plot(table_new_cases_concat['World'],table_new_cases_concat['Open'])
#plt.plot(table_concat['Date'],table_concat['Close'])


# In[ ]:


xval = (table2["Date"][0:1143])
#swag = table2.loc[table2["Date"] == "2020-08-03"]
#xval = list(range(1258))
#print(xval)
yval = table2["Open"][0:1143]
print(yval)

xTrain = xval[0:571]
xTest =  xval[571:1143]
yTrain = yval[0:571]
yTest =  yval[571:1143]

regr = linear_model.LinearRegression()
regr.fit(xTrain, yTrain)
#plt.plot(xval,yval)


# In[221]:


pip install nbzip


# In[8]:


#def sector_splitter(sector):
    #comp

    

Basic_Materials_df = pd.read_csv('Basic_materials_Sector.csv')
Basic_materials_df[0:1256]
Basic_materials_df[1256*1+1:1256*2]
Basic_materials_df[(1256)*2+2:1256*3]
Basic_materials_df[(1256)*3+3:1256*4]


# In[9]:


#12 31 19 Covid China
#1098
#1153 drop
Sectors_names = ['Basic_Materials_Sector', 'Communication_Services_Sector','Consumer_Cyclical_Sector', 'Consumer_Defensive_Sector','Financial_Services_Sector','Real_Estate_Sector','Consumer_Defensive','Healthcare_Sector','Utilities_Sector', 'Energy_Sector']
def biggest_drop(sector):
    max_point_preCovid = max(sector['Open'][1000:1153])
    min_point_covid = min(sector['Open'][1098:1259])
    sector[1000:1259].plot(x ='Date', y='Open', kind = 'line')
    max_point_Covid = max(sector['Open'][1153:1259])
    return max_point_preCovid,min_point_covid,max_point_Covid
  
biggest_drop(Basic_Materials_df)
    


# In[28]:


revenue_df = pd.read_csv('revenue_by_sectors (2).csv')
revenue_df = revenue_df[::-1]
plt.rcParams['figure.figsize'] = [20, 5]
#revenue_df[0:22].plot(x ='date', y='revenue', kind = 'line')
#revenue_df[23:45].plot(x ='date', y='revenue', kind = 'line')
#revenue_df[46:66].plot(x ='date', y='revenue', kind = 'line')
#regression = pd.ols(y=y, x=x)
#regression.summary
#x = np.asarray(x, dtype='string')
revenue_df.plot(x ='date', y='revenue', kind = 'line')
revenue_df['date'] = revenue_df['date'].astype(str)
#revenue_df['date'] = np.asarray(revenue_df['date'], dtype='string')
revenue_df = pd.read_csv('revenue_by_sectors (2).csv')
revenue_df = revenue_df[::-1]
revenue_df = revenue_df[0:22]
sns.regplot(revenue_df['date'],revenue_df['revenue'])


# In[27]:


revenue_df = pd.read_csv('revenue_by_sectors (2).csv')
revenue_df = revenue_df[::-1]
sns.lmplot(x='date', y='revenue', data=revenue_df[0:22], ci=None)


# In[32]:


revenue_df = pd.read_csv('revenue_by_sectors (2).csv')
revenue_df = revenue_df[::-1]
plt.rcParams['figure.figsize'] = [20, 5]
revenue_df[0:22].plot(x ='date', y='revenue', kind = 'line')
plt.savefig('Revenue_Utilities.png', bbox_inches='tight')


# In[ ]:




