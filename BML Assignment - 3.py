#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[20]:


df = pd.read_csv("C:/Users/hp/OneDrive/Desktop/BmlAssignment3.csv")
df


# In[21]:


import math
med_testscore = math.floor(df.test_score.median())
med_testscore


# In[22]:


median_experience = math.floor(df.experience.median())
median_experience


# In[23]:


df.experience = df.experience.fillna(median_experience)
df


# In[24]:


df.test_score = df.test_score.fillna(med_testscore)
df


# In[25]:


reg  = linear_model.LinearRegression()
reg.fit(df[['experience','test_score','interview_score']],df.salary)


# In[26]:


reg.coef_


# In[27]:


reg.intercept_


# In[28]:


reg.predict([[2,9,6]])


# In[29]:


2813.00813008*12+1333.33333333*10+2926.82926829*10+11869.91869918698

