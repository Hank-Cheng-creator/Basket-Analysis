#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install apyori')


# In[174]:


import pandas as pd
from apyori import apriori

pd.set_option('display.max_columns', None)


# In[175]:


df = pd.read_csv("Market_Basket_Optimisation.csv", header=None) #載入原始資料

df.head() #讀取資料中前五筆記錄


# In[176]:


df.fillna(0,inplace=True) #將原始資料中發生NaN的欄位找出並且補上0 (遺漏值處理)
df.head() #再次讀取處理遺漏值後的資料前五筆記錄


# In[177]:


#使用apriori套件必須將資料處理成List型態

transactions = []

for i in range(0,len(df)): #透過for迴圈拜訪df中的每一筆交易記錄
    transactions.append([str(df.values[i,j]) for j in range(0,20) if str(df.values[i,j])!='0'])
    #透過append()針對transactions將拜訪到的資料添加進去，其中i為列、j為欄。
    #由於顧客交易記錄的是商品名稱，因此在添加資料時宜將交易記錄透過str()轉換成文字型態資料。
    #若[i, j]其中某筆記錄發現0，則不予加入transactions list。


# In[178]:


#呼叫apriori()，括號內接收五項參數:
# 1. transactions原始資料來源、2. min_support最小支持度設定、3. min_confidance=0.2最小信賴度設定
# 4. min_lift最小提升度設定、5. min_length商品項目最小交集數量，即每條法則最少需有2個商品。
rules = apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)


# In[179]:


Associaiton_Results = list(rules) #將上述探勘出的法則rules轉換成為list型態資料
Associaiton_Results #顯示探勘到的法則內容


# In[180]:


print(len(Associaiton_Results)) #共有80條法則


# In[181]:


Associaiton_Results[0]  #因為內含frozenset資料，因此無法直接提取內容。


# In[200]:


for rule in Associaiton_Results:
    
    pair1 = rule[2][0][0]
    pair2 = rule[2][0][1]
    
    ante_items = [x for x in pair1]   #因為內含frozenset資料，無法直接提取內容，故使用x for x in語法來拜訪資料。
    conse_items = [x for x in pair2]  #因為內含frozenset資料，無法直接提取內容，故使用x for x in語法來拜訪資料。
    
    print("法則: " + str(ante_items) + "->" + str(conse_items))
    print("支持度: " + str(rule[1]))
    print("信賴度: " + str(rule[2][0][2]))
    print("提升度: " + str(rule[2][0][3]))
    print("=======================================")
    


# In[153]:


arr1 = []

for i in range(10):
    arr1.append(i)

print(arr1)


# In[154]:


arr2 = [i for i in range(10)]

print(arr2)


# In[ ]:




