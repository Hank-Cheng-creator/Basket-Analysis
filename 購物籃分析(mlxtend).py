#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('pip', 'install mlxtend')


# In[37]:


from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules
import pandas as pd


# In[38]:


df1 = pd.read_csv('data.csv', encoding="ISO-8859-1") #讀取原始資料
df1.head()


# In[39]:


df1 = df1[df1.Country == 'France'] #僅挑選法國資料進行後續分析
df1


# In[40]:


df1['Description'] = df1['Description'].str.strip() #將可能的空白移除
df1


# In[41]:


df1 = df1[df1.Quantity >0] #僅挑選商品數量大於0的記錄
df1


# In[42]:


basket = pd.pivot_table(data=df1,index='InvoiceNo',columns='Description',values='Quantity', 
                        aggfunc='sum',fill_value=0)

#繪製樞紐分析表，括號內第一個參數為資料來源、第二個參數為索引欄位、第三個參數為使用欄位、第四個參數為數值、
#第五個參數為數值表達方式，預設為mean平均數，在此設定為sum總和。

basket.head()


# In[24]:


def convert_into_binary(x): #設計一個自訂函式，若x值大於0，則回傳1，否則回傳0。
    if x > 0:
        return 1
    else:
        return 0


# In[43]:


basket_sets = basket.applymap(convert_into_binary) #將basket資料表對應到上述自訂函式予以掃描
basket_sets.head(20)


# In[44]:


basket_sets.drop(columns=['POSTAGE'],inplace=True) #移除POSTAGE欄位
basket_sets.head(40)


# In[45]:


frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

#使用apriori()產生頻繁資料集
#第一個參數為資料來源basket_sets、第二個參數為最小支持度設定成0.07
#第三個參數為是否顯示資料集內容

frequent_itemsets


# In[46]:


rules_mlxtend = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

#使用association_rules()產生關聯法則
#第一個參數為導入頻繁資料集frequent_itemsets、第二個參數為篩選機制設定，此例以lift為主。
#第三個參數為設定lift門檻值，此例設定1。

#leverage 可忽略
#conviction 可忽略

rules_mlxtend.head() 


# In[47]:


rules_mlxtend[ (rules_mlxtend['lift'] >= 4) & (rules_mlxtend['confidence'] >= 0.8) ]

#從所產生的rules_mlxtend關聯法則中提取出lift大於等於4且信賴度大於等於0.8的法則

