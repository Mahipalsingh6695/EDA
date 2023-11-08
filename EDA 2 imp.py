#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("Social_Network_Ads.csv")


# In[3]:


df.head()


# In[4]:


df=df.iloc[:,2:]


# In[5]:


df


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


df.drop("Purchased",axis=1)


# In[8]:


df.shape


# In[9]:


train_test_split(df.drop("Purchased",axis=1),df["Purchased"],test_size=0.25)


# In[10]:


X_train,X_test,y_train,y_test=train_test_split(df.drop("Purchased",axis=1),df["Purchased"],test_size=0.25)


# In[11]:


X_train


# In[12]:


X_test


# In[13]:


y_train


# In[14]:


y_test


# In[15]:


from sklearn.preprocessing import StandardScaler


# In[16]:


scaler=StandardScaler()


# In[17]:


scaler.fit(X_train)


# In[18]:


X_train_sclaed=scaler.transform(X_train)


# In[19]:


X_test_sclaed=scaler.transform(X_test)


# In[20]:


X_train_scaled=pd.DataFrame(X_train_sclaed,columns=X_train.columns)


# In[21]:


X_train


# In[22]:


X_train.describe()


# In[23]:


X_train_scaled


# In[24]:


import numpy as np
np.round(X_train_scaled.describe(),1)


# In[25]:


import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.scatter(X_train['Age'], X_train['EstimatedSalary'])
ax1.set_title("Before Scaling")

ax2.scatter(X_train_scaled['Age'], X_train_scaled['EstimatedSalary'],color='red')
ax2.set_title("After Scaling")

plt.show()


# In[26]:


import seaborn as sns
sns.kdeplot(X_train['EstimatedSalary'])


# In[27]:


import seaborn as sns
sns.kdeplot(X_train_scaled['EstimatedSalary'])


# In[28]:


X_train.describe()


# In[29]:


# min-max

from sklearn.preprocessing import MinMaxScaler

scler_min_max=MinMaxScaler()


# In[30]:


scler_min_max.fit(X_train)


# In[31]:


X_train_min_max=scler_min_max.transform(X_train)


# In[32]:


X_test_min_max=scler_min_max.transform(X_test)


# In[33]:


X_train_min_max = pd.DataFrame(X_train_min_max, columns=X_train.columns)
X_test_min_max = pd.DataFrame(X_test_min_max, columns=X_test.columns)


# In[34]:


np.round(X_train_min_max.describe(),1)


# In[35]:


X_test_min_max


# In[36]:


import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.scatter(X_train['Age'], X_train['EstimatedSalary'])
ax1.set_title("Before Scaling")

ax2.scatter(X_train_min_max['Age'], X_train_min_max['EstimatedSalary'],color='red')
ax2.set_title("After Scaling")

plt.show()


# In[72]:


df3=pd.read_csv("titanic_train.csv")


# In[73]:


df3


# In[40]:


df3=df3[["Age","Fare","SibSp","Survived"]]


# In[41]:


df3


# In[42]:


df3.info()


# In[47]:


df3.isnull().mean()


# In[49]:


df3["Age"].


# In[55]:


X_train,X_test=train_test_split(df3)


# In[56]:


X_test


# In[58]:


X_train


# In[61]:


mean_age=X_train["Age"].mean()


# In[64]:


X_train["Age"].fillna("mean_age")


# In[67]:


X_train["Age"].isnull().sum()


# In[66]:


X_train["Age"].fillna(mean_age).isnull().sum()


# In[68]:


mediann_age=X_train["Age"].median()


# In[69]:


X_train["Age"].fillna(mediann_age)


# In[74]:


df3["Cabin"].mode()


# In[75]:


df3["Cabin"].fillna(df3["Cabin"].mode()[0])


# In[ ]:




