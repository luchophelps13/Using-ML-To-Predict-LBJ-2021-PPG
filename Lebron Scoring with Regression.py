#!/usr/bin/env python
# coding: utf-8

# # Gathering Data

# In[ ]:


# x = The age Lebron turned in that season, y = PPG for that season


# In[195]:


import numpy as np
x = np.array([19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0])


# In[196]:


y = np.array([20.9, 27.2, 31.4, 27.3, 30.0, 28.4, 29.7, 26.7, 27.1, 26.8, 27.1, 25.3, 25.3, 26.4, 27.5, 27.4, 25.7])


# In[197]:


import matplotlib.pyplot as plt

plt.scatter(x, y)

plt.xlabel('Age')
plt.ylabel('Points Per Game')
plt.title("Lebron James' PPG at each Age")
plt.grid()
plt.show()


# # Reshaping & Splitting Data 

# In[46]:


x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
print('x:', x, '\n\n', 'y:', y)


# In[151]:


from sklearn.model_selection import train_test_split 

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.23529411765, random_state=0)


# # Linear Regression

# In[22]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[152]:


regressor.fit(train_x, train_y)


# In[153]:


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)


# In[154]:


y_pred = regressor.predict(test_x)


# In[156]:


df = pd.DataFrame({'Actual': test_y.flatten(), 'Predicted': y_pred.flatten()})
df


# In[157]:


import matplotlib.pyplot as plt

plt.scatter(test_x, test_y, label = 'Actual PPG')
plt.scatter(test_x, y_pred, label = 'Predicted PPG')
plt.xlabel('Age')
plt.ylabel('Points Per Game')
plt.grid()
plt.legend()
plt.show()


# # Metrics

# In[69]:


from sklearn import metrics


# In[158]:


# MAE = |Actual Values - Predicted Values| (Take absolute value)
# MAE is not sensitive to outliers compared to MSE
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, y_pred))  


# In[159]:


#MSE = (Actual Values - Predicted Values)^2
#Most common, but not goof when a single bad value would ruin the prediction
#Very good with outliers

print('Mean Squared Error:', metrics.mean_squared_error(test_y, y_pred))  


# In[160]:


#RMSE = √(Actual Values - Predicted Values)^2

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, y_pred)))


# In[ ]:


# I tried using 3/17 and 5/17 of the columns for test (you usually want to stay between 20-30%) but that led to higher errors


# # Prediction

# In[ ]:


# Predicting 2020-2021 PPG (Assuming MPG, FG%, etc stay relatively the same)


# In[161]:


plt.plot(x, y, label = "Career PPG")
plt.plot(test_x, y_pred, label = "Predicted PPG")
plt.xlabel('Age')
plt.ylabel('PPG')
plt.title("Lebron's PPG Per Season")
plt.grid()
plt.legend(loc = "lower right")
plt.show()


# In[213]:


xnew = np.array([36.0])
xnew = xnew.reshape(-1, 1)

xnew


# In[211]:


x.reshape(-1, 1)


# In[214]:


x_new = np.concatenate((x, xnew))


# In[190]:


ynew = regressor.predict(xnew)


# In[191]:


ynew


# In[185]:


y_new = np.concatenate((y, ynew))

print(y_new)


# In[194]:


plt.scatter(x, y, label = "Career PPG")
plt.plot(x_new, y_new, label = "Predicted PPG in 2020-2021")
plt.xlabel('Age')
plt.ylabel('PPG')
plt.title("Lebron's PPG Per Season")
plt.grid()
plt.legend(loc = "lower right")
plt.show()


# In[ ]:




