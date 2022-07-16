#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
df = pd.read_csv("abcd.csv")
df.head(10)


# In[8]:


import matplotlib.pyplot as plt
plt.scatter(df['distance'],df['price'])


# In[9]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
x = df[['distance']]
y = df['price']
reg = LinearRegression()
reg.fit(x,y)


# In[10]:


reg.predict([[25]])


# In[11]:


reg.coef_


# In[12]:


reg.intercept_


# In[13]:


25*672.22222222-2750.0000000000036


# In[14]:


import pandas as pd
df = pd.read_csv("abcde.csv")
df.head()


# In[15]:


X = df[['distance', 'years']]
y = df['price']


# In[16]:


X


# In[17]:


y


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf


# In[19]:


clf.fit(X_train, y_train)
clf.predict(X_test)


# In[20]:


clf.score(X_test, y_test)


# In[21]:


clf.coef_


# In[22]:


clf.intercept_


# In[23]:


clf.predict([[350,4]])


# In[ ]:




