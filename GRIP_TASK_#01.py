#!/usr/bin/env python
# coding: utf-8

# # GRIP: The Sparks Foundation
# Data Science and Business Analytice Internship

# Author: Molla Mohammad Miah
# 
# Task1: pridiction Using supervised Machine learning

# In this regression task,we predict the percentage of marks that a student is expected to score based upon the number of hours 
# they studied.we use the data available at http://bit.ly/w-data The dataset contains two variable-Hours indicating the number of
# hours of student studies and scorse indication the percentage score he/she received by studying for the corresponding hours.

# In[4]:


#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns


# In[5]:


#loading the dataset
data = pd.read_csv("studen_score.csv")


# In[6]:


print('shape of the dataset is: ',data.shape)
data.head(20)


# In[7]:


#chaecking the missing value in the dataframe
data.isnull().sum()


# In[8]:


data.info()


# In[9]:


data.describe()


# In[24]:


#Plotting the distributio of scores
data.plot(x='Hours',y='Scores',style = 'o')
plt.title("Hours vs Score")
plt.xlabel('Hours')
plt.ylabel('Score')
plt.show()


# In[11]:


data.corr(method= 'pearson')


# In[12]:


data.corr(method='spearman')


# # Linear Regression

# In[13]:


x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


# In[14]:


#splittin data into training and test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[15]:


#Training the algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
print("Training complete")


# In[34]:


line= regressor.coef_*x+regressor.intercept_
#ploting the test data 
plt.scatter(x,y)
plt.plot(x,line);
plt.show()


# In[28]:


#Making predictions
y_pred = regressor.predict(x_test)
print(x_test)


# In[35]:


#compearing actual vs predicted
df = pd.DataFrame({'Actual': y_test,'Predicted':y_pred})
print(df)


# # what will be predicted score if  a student studies for 9.25 hours/day?
# 

# In[32]:


study_hour=[[9.25]]
score_prediction=regressor.predict(study_hour)
print("Number of Hours of studying:",study_hour)
print ("Prediction Score:",score_prediction)


# In[21]:


from sklearn import metrics
print('Mean Absolute Error:',
     metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:




