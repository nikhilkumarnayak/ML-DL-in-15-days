#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
df = pd.read_csv("result.csv")
df.head(10)


# In[74]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.results,train_size=0.8,random_state=10)


# In[75]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


# In[76]:


y_predicted = model.predict(X_test)
y_predicted


# In[77]:


model.score(X_test,y_test)


# In[78]:


from sklearn import datasets
from sklearn import linear_model


# In[79]:


digits = datasets.load_digits()
digits.keys()


# In[80]:


digits.target_names


# In[81]:


digits.data.shape


# In[82]:


digits.images.shape


# In[83]:


digits.images[0]


# In[84]:


import pylab as py
py.matshow(digits.images[0])
py.gray()


# In[85]:


X = digits.data
y = digits.target


# In[86]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)


# In[87]:


digreg = LogisticRegression()


# In[88]:


digreg.fit(X_train, y_train)


# In[89]:


y_pred = digreg.predict(X_test)


# In[90]:


print(metrics.accuracy_score(y_test, y_pred))


# In[91]:


y_pred


# In[ ]:





# In[ ]:




