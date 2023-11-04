#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('spam.csv',encoding='ISO-8859-1')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.size


# In[7]:


df.info()


# In[8]:


df.describe()


# In[11]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[12]:


df


# In[13]:


df=df.rename(columns={'v1':'Target','v2':'Message'})


# In[14]:


df.isnull().sum()


# In[15]:


df.duplicated().sum()


# In[16]:


df.drop_duplicates(keep='first',inplace=True)


# In[17]:


df.duplicated().sum()


# In[18]:


df.size


# In[19]:


from  sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['Target']=encoder.fit_transform(df['Target'])
df['Target']


# In[20]:


df.head()


# In[22]:


plt.pie(df['Target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")


# In[23]:


x=df['Message']
y=df['Target']


# In[25]:


print(x)


# In[26]:


y


# In[27]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=3)


# In[28]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm


# In[29]:


cv=CountVectorizer()


# In[30]:


x_train_cv=cv.fit_transform(x_train)
x_test_cv=cv.transform(x_test)


# In[31]:


print(x_train_cv)


# In[32]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[33]:


lr.fit(x_train_cv,y_train)
prediction_train=lr.predict(x_train_cv)


# In[34]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,prediction_train)*100)


# In[35]:


prediction_test= lr.predict(x_test_cv)


# In[ ]:


from sklearn.metricis import accuracy_score
print(accuracy_scor)

