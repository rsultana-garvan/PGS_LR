#!/usr/bin/env python
# coding: utf-8

# # Cooper 142 SNPs set
# 
# 
# ## Preparation
# 
# ### Import required packages.

# In[1]:


import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning


# ### Read input matrix with genotypes
# 
# The matrix contains the genotypes from AMP-PD/MGRB dataset for 140 SNPs.

# In[2]:


table = pd.read_csv("data/matrix.txt", sep="\t")
table


# ### Distribution of data
# 
# #### Distribution by phenotype
# 
# (0=Control, 1=Case)

# In[3]:


table.groupby('phenotype')['participant_id'].nunique()


# #### Distribution by gender/phenotype

# In[4]:


table.groupby(['gender', 'phenotype'])['participant_id'].nunique()


# #### Distribution by gender/phenotype/inv8_001 genotype

# In[5]:


table.groupby(['gender', 'phenotype', 'inv_genotype'])['participant_id'].nunique()


# ## All participants
# 
# ### Logistic regression model

# In[6]:


pd.set_option('display.max_rows', 150)
X = table[table.columns[5:]]
Y = table['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table.groupby('phenotype')['participant_id'].nunique()


# ### Grid search for 3 hyperparameters

# In[7]:


# parameters = {'C': [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 10, 20, 30],
#              'max_iter': [10, 25, 50, 75, 100, 150, 200, 400, 800, 1600],
#              'l1_ratio': [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}

parameters = {'C': [0.01, 0.02, 0.05],
              'max_iter': [10, 25, 50],
              'l1_ratio': [1, 0.9, 0.8]}

grid_lr = GridSearchCV(lr, parameters, verbose=False, scoring='roc_auc', n_jobs=-1, cv=10)
if not sys.warnoptions:
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
grid_lr.fit(X, Y)


# ### Best estimator

# In[8]:


best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')


# ## Males
# 
# ### Logistic regression model

# In[9]:


table1 = table[table.gender == "M"]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()


# ### Grid search for 3 hyperparameters

# In[10]:


# parameters = {'C': [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 10, 20, 30],
#               'max_iter': [10, 25, 50, 75, 100, 150, 200, 400, 800, 1600],
#               'l1_ratio': [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}

parameters = {'C': [0.01, 0.02],
              'max_iter': [75, 100, 150],
              'l1_ratio': [0.2, 0.1]}

grid_lr = GridSearchCV(lr, parameters, verbose=False, scoring='roc_auc', n_jobs=-1, cv=10)
if not sys.warnoptions:
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
grid_lr.fit(X, Y)


# ### Best estimator

# In[11]:


best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')


# ## Females
# 
# ### Logistic regression model

# In[12]:


table1 = table[table.gender == "F"]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()


# ### Grid search for 3 hyperparameters

# In[13]:


# parameters = {'C': [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 10, 20, 30],
#              'max_iter': [10, 25, 50, 75, 100, 150, 200, 400, 800, 1600],
#              'l1_ratio': [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}

parameters = {'C': [0.005, 0.01],
              'max_iter': [25, 50, 75],
              'l1_ratio': [0.1, 0.2]}

grid_lr = GridSearchCV(lr, parameters, verbose=False, scoring='roc_auc', n_jobs=-1, cv=10)
if not sys.warnoptions:
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
grid_lr.fit(X, Y)


# ### Best estimator

# In[14]:


best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')


# ## NN Males
# 
# ### Logistic regression model

# In[15]:


table1 = table[(table.gender == "M") & (table.inv_genotype=="NN")]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()


# ### Grid search for 3 hyperparameters

# In[16]:


# parameters = {'C': [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 10, 20, 30],
#               'max_iter': [10, 25, 50, 75, 100, 150, 200, 400, 800, 1600],
#               'l1_ratio': [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}

parameters = {'C': [0.05, 0.1, 0.5],
              'max_iter': [10, 25],
              'l1_ratio': [0.7, 0.6, 0.5]}

grid_lr = GridSearchCV(lr, parameters, verbose=False, scoring='roc_auc', n_jobs=-1, cv=10)
if not sys.warnoptions:
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
grid_lr.fit(X, Y)


# ### Best estimator

# In[17]:


best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')


# ## NI Males
# 
# ### Logistic regression model

# In[18]:


table1 = table[(table.gender == "M") & (table.inv_genotype=="NI")]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()


# ### Grid search for 3 hyperparameters

# In[19]:


# parameters = {'C': [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 10, 20, 30],
#               'max_iter': [10, 25, 50, 75, 100, 150, 200, 400, 800, 1600],
#               'l1_ratio': [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}

parameters = {'C': [0.005, 0.01],
              'max_iter': [10, 25, 50],
              'l1_ratio': [0.2, 0.1]}

grid_lr = GridSearchCV(lr, parameters, verbose=False, scoring='roc_auc', n_jobs=-1, cv=10)
if not sys.warnoptions:
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
grid_lr.fit(X, Y)


# ### Best estimator

# In[20]:


best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')


# ## II Males
# 
# ### Logistic regression model

# In[21]:


table1 = table[(table.gender == "M") & (table.inv_genotype=="II")]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()


# ### Grid search for 3 hyperparameters

# In[22]:


# parameters = {'C': [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 10, 20, 30],
#               'max_iter': [10, 25, 50, 75, 100, 150, 200, 400, 800, 1600],
#               'l1_ratio': [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}

parameters = {'C': [0.01, 0.02],
              'max_iter': [50, 75, 100],
              'l1_ratio': [0.2, 0.1]}

grid_lr = GridSearchCV(lr, parameters, verbose=False, scoring='roc_auc', n_jobs=-1, cv=10)
if not sys.warnoptions:
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
grid_lr.fit(X, Y)


# ### Best estimator

# In[23]:


best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')


# ## NN Females
# 
# ### Logistic regression model

# In[24]:


table1 = table[(table.gender == "F") & (table.inv_genotype=="NN")]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()


# ### Grid search for 3 hyperparameters

# In[25]:


# parameters = {'C': [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 10, 20, 30],
#               'max_iter': [10, 25, 50, 75, 100, 150, 200, 400, 800, 1600],
#               'l1_ratio': [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}

parameters = {'C': [0.1, 0.5, 1],
              'max_iter': [10, 25],
              'l1_ratio': [0.7, 0.6, 0.5]}

grid_lr = GridSearchCV(lr, parameters, verbose=False, scoring='roc_auc', n_jobs=-1, cv=10)
if not sys.warnoptions:
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
grid_lr.fit(X, Y)


# ### Best estimator

# In[26]:


best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')


# ## NI Females
# 
# ### Logistic regression model

# In[27]:


table1 = table[(table.gender == "M") & (table.inv_genotype=="NI")]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()


# ### Grid search for 3 hyperparameters

# In[28]:


# parameters = {'C': [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 10, 20, 30],
#               'max_iter': [10, 25, 50, 75, 100, 150, 200, 400, 800, 1600],
#               'l1_ratio': [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}

parameters = {'C': [0.005, 0.01],
              'max_iter': [10, 25],
              'l1_ratio': [0.2, 0.1]}

grid_lr = GridSearchCV(lr, parameters, verbose=False, scoring='roc_auc', n_jobs=-1, cv=10)
if not sys.warnoptions:
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
grid_lr.fit(X, Y)


# ### Best estimator

# In[29]:


best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')


# ## II Females
# 
# ### Logistic regression model

# In[30]:


table1 = table[(table.gender == "F") & (table.inv_genotype=="II")]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()


# ### Grid search for 3 hyperparameters

# In[31]:


# parameters = {'C': [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 10, 20, 30],
#               'max_iter': [10, 25, 50, 75, 100, 150, 200, 400, 800, 1600],
#               'l1_ratio': [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}

parameters = {'C': [1, 10, 20],
              'max_iter': [1600, 3200],
              'l1_ratio': [1, 0.9]}

grid_lr = GridSearchCV(lr, parameters, verbose=False, scoring='roc_auc', n_jobs=-1, cv=10)
if not sys.warnoptions:
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
grid_lr.fit(X, Y)


# ### Best estimator

# In[32]:


best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')

