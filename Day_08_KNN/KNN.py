#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
dataset = pd.read_csv('Car.csv')
dataset.head(10)


# In[49]:


from matplotlib import pyplot as plt
plt.scatter(dataset.Age,dataset['Income'])


# In[51]:


df1 = dataset[dataset.Car==0]
df2 = dataset[dataset.Car==1]
plt.scatter(df1.Age,df1['Income'],color='green')
plt.scatter(df2.Age,df2['Income'],color='red')


# In[3]:


X = dataset[['Age','Income']]
y = dataset[['Car']]


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)


# In[32]:


X_train


# In[33]:


X_test


# In[34]:


y_train


# In[35]:


y_test


# In[45]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X_train, y_train)


# In[46]:


print(classifier.predict([[33,149000]]))


# In[47]:


classifier.score(X_test,y_test)


# In[ ]:




