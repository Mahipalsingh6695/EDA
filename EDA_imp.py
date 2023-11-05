#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install seaborn==0.11.2')


# In[2]:


import pandas as pd 
df=pd.read_csv("titanic_train.csv")


# In[3]:


import os, sys
import pandas as pd


# In[4]:


import seaborn as sns


# # 1. size of data

# In[5]:


df.shape


# In[6]:


df.size


# In[7]:


df.memory_usage(deep=True)


# # 2. how the data looks likes

# In[8]:


df.sample(5)


# In[9]:


df.head()


# # 3. what are the dtype of col
# 

# In[10]:


df.dtypes


# In[11]:


df.info()


# In[12]:


df.describe().T


# In[13]:


df.isnull()


# In[14]:


df.isnull().sum()


# In[15]:


df.isnull().sum().sum()


# In[16]:


df.duplicated().sum()


# In[17]:


df[df.duplicated()]


# In[18]:


df.nunique()


# In[19]:


df.corr()


# In[20]:


corr_mat=df.corr()


# In[21]:


import seaborn as sns


# In[22]:


sns.heatmap(corr_mat,annot=True)


# ## univaritae analysis
# ## bivariate analysis
# ## multivariate analysis

# In[23]:


df.head()


# In[24]:


df.columns


# In[25]:


## categeric variable
## numeric variable


# In[26]:


cat_feature=[column for column in df.columns if df[column].dtype=="O"]


# In[27]:


num_feature=[column for column in df.columns if df[column].dtype!="O"]


# In[28]:


df[cat_feature]


# In[29]:


df[num_feature]


# ## categorical column

# In[30]:


import seaborn as sns


# In[31]:


## 1. count plot


# In[32]:


sns.histplot(df["Embarked"])


# In[33]:


sns.histplot(df["Sex"])


# In[34]:


sns.histplot(df["Pclass"])


# In[35]:


df["Sex"].value_counts().plot(kind='bar')


# In[36]:


## 2. pie plot


# In[37]:


df["Sex"].value_counts().plot(kind='pie',autopct='%.2f')


# In[38]:


df["Pclass"].value_counts().plot(kind='pie',autopct='%.2f')


# ## numerical column

# In[39]:


import matplotlib.pyplot as plt


# In[40]:


plt.hist(df["Age"])


# In[41]:


plt.hist(df["Age"],bins=5)


# In[42]:


plt.hist(df["Age"],bins=20)


# In[43]:


sns.distplot(df["Age"],bins=20)


# ## boxplot

# In[44]:


sns.boxplot(df["Age"])


# In[45]:


df["Age"].min


# In[46]:


df["Age"].min()


# In[47]:


df["Age"].max()


# In[48]:


df["Age"].mean()


# In[49]:


df["Age"].median()


# In[50]:


df["Age"].skew()


# In[51]:


1-df["Age"].skew()


# ## bivariate analysis

# In[52]:


#categorical
#numerical 


# In[53]:


# X-> categorical Y-> categorical

# X-> numerical   Y-> numeircal

# X-> catgorical  Y-> numerical


# In[54]:


tips=pd.read_csv("tips.csv")


# In[55]:


flight=pd.read_csv("flights.csv")


# In[56]:


iris=pd.read_csv("iris.csv")


# In[57]:


tips


# In[58]:


# bivariate


# In[59]:


sns.scatterplot(tips["total_bill"],tips["tip"])


# In[60]:


# multivariate 


# In[61]:


sns.scatterplot(tips["total_bill"],tips["tip"],hue=tips["sex"],style=tips["smoker"])


# ## 2. bar plot (X-> numeric variable y-> categorical variable)

# In[62]:


sns.barplot(df["Pclass"],df["Age"],hue=df["Sex"])


# ## 2. box plot (X-> numeric variable y-> categorical variable)

# In[63]:


sns.boxplot(df["Sex"],df["Age"],hue=df["Survived"])


# In[64]:


df[df["Survived"]==0]


# In[65]:


df[df["Survived"]==0]["Age"].max()


# In[66]:


df[df["Survived"]==0]["Age"].min()


# In[67]:


df[df["Survived"]==1]["Age"].max()


# In[68]:


df[df["Survived"]==1]["Age"].min()


# In[69]:


sns.distplot(df[df["Survived"]==0]["Age"],hist=False)
sns.distplot(df[df["Survived"]==1]["Age"],hist=False)


# ## heatmap plot (X-> categoricla variable y-> categorical variable)

# In[70]:


df.head()


# In[71]:


df["Pclass"]


# In[72]:


df["Survived"]


# In[73]:


pd.crosstab(df["Pclass"],df["Survived"])


# In[74]:


sns.heatmap(pd.crosstab(df["Pclass"],df["Survived"]),annot=True)


# In[75]:


iris.head()


# In[76]:


sns.pairplot(iris,hue="species")


# ## Lineplot()

# In[77]:


sns.lineplot(tips["total_bill"],tips["tip"])


# In[78]:


sns.scatterplot(tips["total_bill"],tips["tip"])


# In[79]:


flight


# In[80]:


sns.lineplot(flight["Month"],flight["VX"])


# In[81]:


pip install ydata-profiling


# In[86]:


from ydata_profiling import ProfileReport


# In[88]:


profile=ProfileReport(df,title="pandas profiling report")


# In[89]:


profile.to_file("your_report.html")


# In[ ]:





# In[ ]:




