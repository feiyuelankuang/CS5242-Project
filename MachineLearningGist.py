#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import os
data = []
directory = "/Users/Bumblebee/Desktop/Y4S1/CS5242/Project/gist/train_gist"
for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        tempFileName = "/Users/Bumblebee/Desktop/Y4S1/CS5242/Project/gist/train_gist/" + filename
        value =  np.load(tempFileName)
        value = np.concatenate(value)
        data.append(value)


# In[3]:


df = pd.DataFrame(data)


# In[4]:


df.shape


# In[5]:


labels = pd.read_csv("/Users/Bumblebee/Desktop/Y4S1/CS5242/Project/gist/train_kaggle.csv")


# In[6]:


labels = labels.drop(labels.columns[[0]], axis = 1)


# In[7]:


labels.shape


# In[9]:


from sklearn.preprocessing import StandardScaler

stdScaler = StandardScaler()
df_scaled = stdScaler.fit_transform(df)


# In[13]:


df_scaled = pd.DataFrame(df_scaled)


# In[14]:


combineddf = pd.concat([df_scaled, labels], axis=1)


# In[15]:


combineddf.shape


# In[16]:


from sklearn.model_selection import train_test_split

trainSet, testSet = train_test_split(combineddf, test_size = 0.2, random_state = 13)


# In[17]:


dfTrain = trainSet.drop(trainSet.columns[[512]], axis = 1)
dfLabel = trainSet[trainSet.columns[[512]]].copy()

dfTest = testSet.drop(testSet.columns[[512]], axis = 1)
dfTestLabel = testSet[testSet.columns[[512]]].copy()


# In[18]:


dfLabel.shape


# In[19]:


from sklearn.ensemble import RandomForestClassifier # Ensemble training (Need to check time)
from sklearn.model_selection import KFold #KFold for accuracy prediction

kfold = KFold(n_splits=5, random_state=13)
forestClassifier = RandomForestClassifier(random_state = 13).fit(dfTrain, dfLabel)


# In[20]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
accuracyScoresRF = cross_val_score(forestClassifier, dfTrain, dfLabel, cv=kfold, scoring = 'roc_auc')
accuracyScoresRF


# In[ ]:


from sklearn.svm import SVC

modelSVCLinear = SVC(kernel="linear", probability=True)
modelSVCLinear.fit(dfTrain, dfLabel)


# In[44]:


from sklearn.tree import DecisionTreeClassifier 
modelDecisiontree = DecisionTreeClassifier(max_depth = 2)

accuracyScoresDTC = cross_val_score(modelDecisiontree, dfTrain, dfLabel, cv=kfold, scoring = 'roc_auc')
accuracyScoresDTC


# In[54]:


yPred = forestClassifier.predict(dfTest)
print(yPred)
from sklearn.metrics import roc_auc_score
roc_auc_score(dfTestLabel, yPred)


# In[43]:


labels


# In[ ]:




