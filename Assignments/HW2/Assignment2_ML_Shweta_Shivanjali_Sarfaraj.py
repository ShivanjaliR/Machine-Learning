#!/usr/bin/env python
# coding: utf-8

# In[326]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function Algo_SMO takes the dataset, labels, C, tolerance, iteration as input and provides w and b as a output. 

# In[327]:


def Algo_SMO(dataset_x, labels_y, C, tolerance, max_iter):
       X = np.asmatrix(dataset_x);
       Y = np.asmatrix(labels_y).T
       l,n = np.shape(X)
       # b is offset and initially set to zero. alpha_1 is a lagrange multipliers. Initialize alpha.
       b = 0;
       alpha_1 = np.asmatrix(np.zeros((l,1)))
       iter1 = 0
        # The error Ei is calculated and if the error is large then the alpha corresponding to this data 
        # point is optimized
       while (iter1 < max_iter):
              changed_alpha = 0
              for i in range(l):
                    #calculated error denoted by Ei
                     fXi = float(np.multiply(alpha_1,Y).T*(X*X[i,:].T)) + b
                     Ei = fXi - float(Y[i])
                    #choose the value of alpha for which maximum error value of error occurs. This the alpha to be optimized
                     if ((Y[i]*Ei < -tolerance) and (alpha_1[i] < C)) or ((Y[i]*Ei > tolerance) and (alpha_1[i] > 0)):
                           # Randomly select the second alpha
                           j = i
                           while (j==i):
                            j = int(np.random.uniform(0,l))
                           # Calculate the error for second alpha
                           fXj = float(np.multiply(alpha_1,Y).T*(X*X[j,:].T)) + b
                           Ej = fXj - float(Y[j])
                           # Store the value of alpha for computation
                           alphai_old = alpha_1[i].copy();
                           alphaj_old= alpha_1[j].copy();
                           # compute lower and higher value for maintaining the values inside the bounds
                           if (Y[i] != Y[j]):
                                  Low = max(0, alpha_1[j] - alpha_1[i])
                                  High = min(C, C + alpha_1[j] - alpha_1[i])
                           else:
                                  Low = max(0, alpha_1[j] + alpha_1[i] - C)
                                  High = min(C, alpha_1[j] + alpha_1[i])
                           # if L = H the continue to next 
                           if Low==High:
                                  continue
                           # compute eta
                           k = 2.0 * X[i,:]*X[j,:].T - X[i,:]*X[i,:].T - X[j,:]*X[j,:].T
                           # if eta >= 0 then continue to next i
                           if k >= 0:
                                  continue
                           # compute new value for alphas j
                           alpha_1[j] = alpha_1[j] - Y[j]*(Ei - Ej)/k
                           # clip new value for alphas j
                           if alpha_1[j] > High:
                                alpha_1[j] = High
                           if Low > alpha_1[j]:
                                alpha_1[j] = Low
                           # if |alphasj - alphasold| < 0.00001 then continue to next i
                           if (abs(alpha_1[j] - alphaj_old) < 0.00001):
                                  continue
                           # determine value for alphas i
                           alpha_1[i] = alpha_1[i] + Y[j]*Y[i]*(alphaj_old - alpha_1[j])
                           # compute b1 and b2
                           b1 = b - Ei- Y[i]*(alpha_1[i]-alphai_old)*X[i,:]*X[i,:].T - Y[j]*(alpha_1[j]-alphaj_old)*X[i,:]*X[j,:].T
                                      
                           b2 = b - Ej- Y[i]*(alpha_1[i]-alphai_old)*X[i,:]*X[j,:].T - Y[j]*(alpha_1[j]-alphaj_old)*X[j,:]*X[j,:].T
                                    
                           # computation of b
                           if (0 < alpha_1[i]) and (C > alpha_1[i]):
                                  b = b1
                           elif (0 < alpha_1[j]) and (C > alpha_1[j]):
                                  b = b2
                           else:
                                  b = (b1 + b2)/2.0                      
                           changed_alpha += 1
                     if (changed_alpha == 0): iter1 += 1
                     else: iter1 = 0
       return b,alpha_1


# In[328]:


df = pd.read_csv("smo_assignment.csv")


# In[329]:


df.head()


# In[330]:


y= df.iloc[:,-1]


# In[331]:


y.head()


# In[332]:


x = df.iloc[:,0:2]


# In[333]:


x.head()


# In[334]:


b,alpha_1 = Algo_SMO(dataset_x = x, labels_y = y, C = .6, tolerance =.001, max_iter=5)


# In[335]:


b


# In[336]:


alpha_1


# In[337]:


def computeW(alpha_1, data_X, label_Y):
      X = np.asmatrix(data_X)
      Y = np.asmatrix(label_Y).T
      l,n = np.shape(X)
      w = np.zeros((n,1))
      for i in range(l):
            w += np.multiply(alpha_1[i]*Y[i],X[i,:].T)
      return w


# In[338]:


w = computeW(alpha_1 = alpha_1, data_X=x,label_Y = y)


# In[339]:


print(w)


# In[340]:


# Test for classification
points = [450,2]
predicted_value = np.dot(w.T,points)+b
if predicted_value >= 0:
    class_1 = predicted_value
    print("Predicted value of point belongs to class_1:",class_1)
else:
    class_2 = predicted_value
    print("Predicted value of point belongs to class_2:",class_2)
    


# In[ ]:




