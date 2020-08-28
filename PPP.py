#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[ ]:





# In[ ]:





# In[ ]:


df = pd.ExcelFile('PPPdataAL.xlsx')   
PPP_Data_All = pd.read_excel(df, 'PPP Data small')
Bigloans_All = pd.read_excel(df, 'Bigloans')
 = pd.read_excel(df, 'Analysis')
Analysis_All
state_list = ["AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
for state in state_list:
    file = 'PPPdata' + state + ".xlsx"
    print(file)
    df = pd.ExcelFile(file)
    df1 = pd.read_excel(df, 'PPP Data small')
    PPP_Data_All = PPP_Data_All.append(df1)
    df2 = pd.read_excel(df, 'Bigloans')
    Bigloans_All =  Bigloans_All.append(df2)
    df3 = pd.read_excel(df, 'Analysis')
    Analysis_All = Analysis_All.append(df3)
    #return(PPP_Data_All,Bigloans_All,Analysis_All)
    


# In[122]:


Analysis_All


# In[ ]:


#include the summary industry numbers of summed establishments and number of employees. And not the % of QCEW coverge. 


# In[2]:


CA_df = pd.ExcelFile('PPPdataCA.xlsx')   
PPP_Data_CA = pd.read_excel(CA_df, 'PPP Data small')
Bigloans_CA = pd.read_excel(CA_df, 'Bigloans')


# In[3]:


PPP_Data_CA


# In[ ]:




