#!/usr/bin/env python
# coding: utf-8

# In[43]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
df = pd.read_csv("Book.csv")
df.head(50)


# In[44]:


plt.scatter(df.rollno,df['marks'])
plt.xlabel('rollno')
plt.ylabel('marks')


# In[45]:


km = KMeans(n_clusters=3)
predicted = km.fit_predict(df[['rollno','marks']])
predicted


# In[46]:


df['cluster']=predicted
df.head()


# In[47]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.rollno,df1['marks'],color='green')
plt.scatter(df2.rollno,df2['marks'],color='red')
plt.scatter(df3.rollno,df3['marks'],color='black')
plt.xlabel('rollno')
plt.ylabel('marks')


# In[67]:


scale = MinMaxScaler()

scale.fit(df[['marks']])
df['marks'] = scale.transform(df[['marks']])

scale.fit(df[['rollno']])
df['rollno'] = scale.transform(df[['rollno']])
df


# In[63]:


km = KMeans(n_clusters=3)
km.fit(df[['rollno','marks']])
predicted = km.predict(df[['rollno','marks']])
predicted


# In[64]:


df = df.drop(['cluster'], axis='columns')

df['cluster']=predicted
df.head()


# In[65]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.rollno,df1['marks'],color='green')
plt.scatter(df2.rollno,df2['marks'],color='red')
plt.scatter(df3.rollno,df3['marks'],color='black')
plt.xlabel('rollno')
plt.ylabel('marks')


# In[66]:


km.cluster_centers_


# In[ ]:





# In[ ]:




