#!/usr/bin/env python
# coding: utf-8

# In[11]:


from sklearn.datasets import load_iris
dataset = load_iris()
dataset


# In[13]:


df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
df.head()


# In[14]:


df['target'] = dataset.target
df.head()


# In[15]:


from sklearn.model_selection import train_test_split
X = df.drop(['target'], axis='columns')
y = df.target


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)


# In[22]:


print(X_train)
print(y_train)
print(X_test)
print(y_test)


# In[23]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


# In[24]:


classifier.predict(X_test)


# In[25]:


classifier.score(X_test,y_test)


# In[27]:


classifier.predict([[6.7, 3.0, 5.2, 2.3]])


# In[ ]:




