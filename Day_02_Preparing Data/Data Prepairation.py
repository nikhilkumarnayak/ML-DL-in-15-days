#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd
data = pd.read_csv("/Users/shridhar/Desktop/purchase.csv")
data


# In[95]:


data.describe()


# In[96]:


#mean
data['Ball'] = data['Ball'].fillna(data['Ball'].mean())
data


# In[97]:


#median
data['Bat'] = data['Bat'].fillna(data['Bat'].median())
data


# In[98]:


#standard deviatiom  
data['apple'] = data['apple'].fillna(data['apple'].std())
data


# In[99]:


#min  
data['Orange'] = data['Orange'].fillna(data['Orange'].min())
data


# In[100]:


#max  
data['Price'] = data['Price'].fillna(data['Price'].max())
data


# In[119]:


from pandas import read_csv
from numpy import set_printoptions
from sklearn import preprocessing
data = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
name = ["A","B","C","D","E","F","G","H","I"]
a = read_csv (data, names=name)
a.describe()


# In[117]:


scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
rescaled = scaler.fit_transform(a)
set_printoptions(precision=2)
rescaled


# In[118]:


from sklearn.preprocessing import StandardScaler
data_scaler = StandardScaler().fit(a)
data_rescaled = data_scaler.transform(a)
data_rescaled


# In[111]:


from sklearn.preprocessing import Binarizer
binary = Binarizer(threshold=0.5)
binary1 = binary.transform(a)
binary1

