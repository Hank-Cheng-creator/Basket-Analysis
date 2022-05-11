#!/usr/bin/env python
# coding: utf-8

# In[7]:


from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd


# In[6]:


data = [['Monitor', 'Mouse', 'Keyboard', 'Notebook', 'Desktop', 'Speaker'],
['Wireless mouse', 'Mouse', 'Keyboard', 'Notebook', 'Desktop', 'Speaker'],
['Monitor', 'Adapter', 'Notebook', 'Desktop'],
['Monitor', 'Microphone', 'Earphone', 'Notebook', 'Speaker'],
['Earphone', 'Mouse', 'Mouse', 'Notebook', 'Headphone', 'Desktop']]

#本練習題原始交易資料，以List型態資料來表達。


# In[15]:


te = TransactionEncoder()
tranfer_array = te.fit(data).transform(data)

#將原始交易資料轉換成DataFrame格式並以布林值顯示，其中fit()為比對、transform()為轉換。

tranfer_array  


# In[17]:


te.columns_


# In[21]:


import pandas as pd
df = pd.DataFrame(tranfer_array, columns=te.columns_)
df #將上述布林陣列轉換成DataFrame以利判讀資料

""" data = [['Monitor', 'Mouse', 'Keyboard', 'Notebook', 'Desktop', 'Speaker'],
['Wireless mouse', 'Mouse', 'Keyboard', 'Notebook', 'Desktop', 'Speaker'],
['Monitor', 'Adapter', 'Notebook', 'Desktop'],
['Monitor', 'Microphone', 'Earphone', 'Notebook', 'Speaker'],
['Earphone', 'Mouse', 'Mouse', 'Notebook', 'Headphone', 'Desktop']] """


# In[22]:


from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules


# In[23]:


apriori(df, min_support=0.6) #調用apriori()以便產生頻繁資料集，最小支持度設定為0.6。


# In[25]:


frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True) 
frequent_itemsets

#透過use_colnames=True參數顯示出頻繁資料集名稱


# In[36]:


rules = frequent_itemsets[frequent_itemsets['support']>=0.8]
rules


# In[37]:


rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
rules


# In[38]:


rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.2)
rules


# In[39]:


rules


# In[41]:


rules[rules['antecedents']=={'Notebook','Desktop'}]

#什麼商品適合跟筆電與桌機一起促銷 (無關合理性)


# In[ ]:




