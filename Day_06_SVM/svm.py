#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
iris


# In[1]:


df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[2]:


df['target'] = iris.target
df.head()


# In[9]:


from sklearn.model_selection import train_test_split
X = df.drop(['target'], axis='columns')
y = df.target


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=8)


# In[29]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)


# In[30]:


model.score(X_test, y_test)


# In[24]:


model.predict([[5.1,3.7,2.5,0.9]])


# In[ ]:




