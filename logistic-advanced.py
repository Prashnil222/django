#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[2]:


# data = np.loadtxt('cricket_data.txt',delimiter = ',',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
# # y = np.loadtxt('ex1data1.txt',delimiter=",",usecols=(1,))
# m = len(data)
# data1 = np.hstack((data,np.ones((m,1))))
# #np.savetxt("ex1data2.txt", data1,fmt="%2.5f")
# X = np.loadtxt('ex1data2.txt',usecols=[19,0,10,11])
# y = np.loadtxt('ex1data2.txt',usecols=[18])
# print(X)


# In[4]:


X = np.loadtxt('ten_over.txt',usecols=[4,0,1,2])
y = np.loadtxt('ten_over.txt',usecols=[3])
print(X,len(X))


# In[5]:


from sklearn.linear_model import LogisticRegression


# In[7]:


logisticRegr = LogisticRegression()


# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[9]:


#x_train


# In[10]:


#y_train


# In[11]:


logisticRegr.fit(x_train,y_train)


# In[20]:


y_pred = logisticRegr.predict(x_test)
print(y_pred)
print(y_test)


# In[21]:


#x_test


# In[22]:


#y_test


# In[29]:


targets = np.array([[1,200,104,1]])
logisticRegr.predict(targets)


# In[24]:


score = logisticRegr.score(x_test, y_test)
print(score)


# In[25]:


confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[26]:


print(classification_report(y_test, y_pred))


# In[72]:


parameters = logisticRegr.coef_


# In[28]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
logit_roc_auc = roc_auc_score(y_test, logisticRegr.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, logisticRegr.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[73]:


'''print(parameters)


# In[74]:


th = np.array([0.13640112,-0.09403004,0.04075078,0.13683161,-0.36898452])
print(th.shape)


# In[75]:


targets = np.array([1,200,149,50,6])
print(targets.shape)


# In[76]:


val= targets@th
val


# In[77]:


print(round(float(Sigmoid(targets@th)),3))


# In[21]:


np.array([1,2,3])@np.array([1, 2, 3])


# In[104]:


a = np.array([1,2,3])


# In[107]:


a.shape


# In[ ]:

'''


