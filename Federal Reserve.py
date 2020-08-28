#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install dash==1.15.0')


# In[10]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[9]:


Federal_Reserve_df = pd.ExcelFile('NY_FED_HHD_C_Report_2020Q2.xlsx')
df1 = pd.read_excel(Federal_Reserve_df, 'Page 3 Data')
df2 = pd.read_excel(Federal_Reserve_df, 'Page 20 Data')

df2


# In[3]:


Federal_Reserve_20_df = pd.read_csv('FederalReserve20.csv')
Federal_Reserve_20_df.tail(40)
#plt.style.use(['dark_background', 'presentation'])


# In[13]:


plt.rcParams['figure.figsize'] = [25, 5]
Federal_Reserve_20_df.plot(kind='scatter',x='quarter',y='18-29',color='blue')
plt.show()
Federal_Reserve_20_df.plot(kind='scatter',x='quarter',y='30-39',color='blue')
plt.show()
Federal_Reserve_20_df.plot(kind='scatter',x='quarter',y='40-49',color='blue')
plt.show()
Federal_Reserve_20_df.plot(kind='scatter',x='quarter',y='50-59',color='blue')
plt.show()
Federal_Reserve_20_df.plot(kind='scatter',x='quarter',y='60-69',color='blue')
plt.show()
Federal_Reserve_20_df.plot(kind='scatter',x='quarter',y='70',color='blue')
plt.show()


# In[ ]:





# In[18]:


Federal_Reserve_20_df.plot(kind='scatter',x='quarter',y='18-29',color='blue')
plt.tick_params(axis='both', labelsize=0, length = 0)
plt.savefig('18-29.png')
plt.show()
Federal_Reserve_20_df.plot(kind='scatter',x='quarter',y='30-39',color='blue')
plt.tick_params(axis='both', labelsize=0, length = 0)
plt.savefig('30-39.png')
plt.show()
Federal_Reserve_20_df.plot(kind='scatter',x='quarter',y='40-49',color='blue')
plt.tick_params(axis='both', labelsize=0, length = 0)
plt.savefig('40-49.png')
plt.show()
Federal_Reserve_20_df.plot(kind='scatter',x='quarter',y='50-59',color='blue')
plt.tick_params(axis='both', labelsize=0, length = 0)
plt.savefig('50-59.png')
plt.show()
Federal_Reserve_20_df.plot(kind='scatter',x='quarter',y='60-69',color='blue')
plt.tick_params(axis='both', labelsize=0, length = 0)
plt.savefig('60-69.png')
plt.show()
Federal_Reserve_20_df.plot(kind='scatter',x='quarter',y='70',color='blue')
plt.tick_params(axis='both', labelsize=0, length = 0)
plt.savefig('70.png')
plt.show()


# In[ ]:




