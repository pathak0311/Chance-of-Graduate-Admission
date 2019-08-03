#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[61]:


cd 'Downloads/graduate-admissions/'


# In[62]:


data = pd.read_csv('Admission_Predict.csv')


# In[63]:


data.describe()


# In[64]:


data.info()


# In[65]:


X = data.iloc[:, 1:-1]
y = data.iloc[:, 8:]


# In[66]:


X.info()


# In[67]:


y.info()


# In[68]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[69]:


from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0, 1))
X_train[X_train.columns] = scalerX.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = scalerX.transform(X_test[X_test.columns])


# In[70]:


from sklearn.linear_model import LinearRegression
lnr_rgsr = LinearRegression()
lnr_rgsr.fit(X_train, y_train)


# In[71]:


print(lnr_rgsr.score(X_test, y_test))
y_pred_lr = lnr_rgsr.predict(X_test)


# In[78]:


from sklearn.ensemble import RandomForestRegressor
rf_rgsr = RandomForestRegressor(n_estimators = 10)
rf_rgsr.fit(X_train, y_train)


# In[79]:


y_pred = rf_rgsr.predict(X_test)


# In[80]:


rf_rgsr.score(X_test, y_test)


# In[75]:


import seaborn as sns
sns.heatmap(data.corr(), annot = True)


# In[76]:


sns.pairplot(data)


# In[77]:


from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred_lr))

