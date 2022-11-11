---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# Cooper 142 SNPs set


## Preparation

### Import required packages.

```{code-cell}
import os, sys, warnings
import numpy as np
import pandas as pd
import statistics as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.exceptions import ConvergenceWarning
```

### Read input matrix with genotypes

The matrix contains the genotypes from AMP-PD/MGRB dataset for 140 SNPs.

```{code-cell}
table = pd.read_csv("data/matrix.txt", sep="\t")
table
```

### Distribution of data

#### Distribution by phenotype

(0=Control, 1=Case)

```{code-cell}
table.groupby('phenotype')['participant_id'].nunique()
```

#### Distribution by gender/phenotype

```{code-cell}
table.groupby(['gender', 'phenotype'])['participant_id'].nunique()
```

#### Distribution by gender/phenotype/inv8_001 genotype

```{code-cell}
table.groupby(['gender', 'phenotype', 'inv_genotype'])['participant_id'].nunique()
```


## All participants

### Logistic regression model

```{code-cell}
pd.set_option('display.max_rows', 150)
X = table[table.columns[5:]]
Y = table['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table.groupby('phenotype')['participant_id'].nunique()
```

### Grid search for 3 hyperparameters

```{code-cell}
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
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score:.4f}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best AUC score: {grid_lr.best_score_:.4f}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')
```

### 5-fold cross validation (10 times)

```{code-cell}
name = "all"
AUCs = list()
bp = grid_lr.best_params_
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
clf = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

count = 0
for train_index, test_index in rkf.split(X):
    count = count + 1

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred)
    AUCs.append(auc)

    X_header = np.array(X_train.columns)
    data_array = np.vstack((X_header, clf.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{count}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

# Fit predictor to statistically significant features (just once!!!)
clf.fit(X, Y)
y_pred = clf.predict_proba(X)[:,1]

# This in-sample AUC should be better than the AUCs from the repeated cross-validation
auc = roc_auc_score(Y, y_pred)

#AUC results from the 50 predictors
AUC_out = pd.DataFrame(AUCs, columns=['AUC'])
AUC_out.to_csv(f"data/{name}_AUCs.txt", sep='\t',index=False)

AUC_std= st.stdev(AUCs)
AUC_mean= st.mean(AUCs)

print(f'In-Sample AUC: {auc:.4f}')
print(f'MeanCV AUC: {AUC_mean:.4f}')
print(f'Standard Deviation CV AUC: {AUC_std:.4f}')
```

## Males

### Logistic regression model

```{code-cell}
table1 = table[table.gender == "M"]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()
```

### Grid search for 3 hyperparameters

```{code-cell}
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
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score:.4f}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_:.4f}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')
```

### 5-fold cross validation (10 times)

```{code-cell}
name = "M"
AUCs = list()
bp = grid_lr.best_params_
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
clf = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

count = 0
for train_index, test_index in rkf.split(X):
    count = count + 1

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred)
    AUCs.append(auc)

    X_header = np.array(X_train.columns)
    data_array = np.vstack((X_header, clf.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{count}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

# Fit predictor to statistically significant features (just once!!!)
clf.fit(X, Y)
y_pred = clf.predict_proba(X)[:,1]

# This in-sample AUC should be better than the AUCs from the repeated cross-validation
auc = roc_auc_score(Y, y_pred)

#AUC results from the 50 predictors
AUC_out = pd.DataFrame(AUCs, columns=['AUC'])
AUC_out.to_csv(f"data/{name}_AUCs.txt", sep='\t',index=False)

AUC_std= st.stdev(AUCs)
AUC_mean= st.mean(AUCs)

print(f'In-Sample AUC: {auc:.4f}')
print(f'MeanCV AUC: {AUC_mean:.4f}')
print(f'Standard Deviation CV AUC: {AUC_std:.4f}')
```


## Females

### Logistic regression model

```{code-cell}
table1 = table[table.gender == "F"]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()
```

### Grid search for 3 hyperparameters

```{code-cell}
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
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score:.4f}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_:.4f}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')
```

### 5-fold cross validation (10 times)

```{code-cell}
name = "F"
AUCs = list()
bp = grid_lr.best_params_
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
clf = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

count = 0
for train_index, test_index in rkf.split(X):
    count = count + 1

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred)
    AUCs.append(auc)

    X_header = np.array(X_train.columns)
    data_array = np.vstack((X_header, clf.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{count}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

# Fit predictor to statistically significant features (just once!!!)
clf.fit(X, Y)
y_pred = clf.predict_proba(X)[:,1]

# This in-sample AUC should be better than the AUCs from the repeated cross-validation
auc = roc_auc_score(Y, y_pred)

#AUC results from the 50 predictors
AUC_out = pd.DataFrame(AUCs, columns=['AUC'])
AUC_out.to_csv(f"data/{name}_AUCs.txt", sep='\t',index=False)

AUC_std= st.stdev(AUCs)
AUC_mean= st.mean(AUCs)

print(f'In-Sample AUC: {auc:.4f}')
print(f'MeanCV AUC: {AUC_mean:.4f}')
print(f'Standard Deviation CV AUC: {AUC_std:.4f}')
```


## NN

### Logistic regression model

```{code-cell}
table1 = table[table.inv_genotype=="NN"]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()
```

### Grid search for 3 hyperparameters

```{code-cell}
# parameters = {'C': [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 10, 20, 30],
#              'max_iter': [10, 25, 50, 75, 100, 150, 200, 400, 800, 1600],
#              'l1_ratio': [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}

parameters = {'C': [0.01, 0.02, 0.05],
              'max_iter': [10, 25],
              'l1_ratio': [0.5, 0.4, 0.3]}

grid_lr = GridSearchCV(lr, parameters, verbose=False, scoring='roc_auc', n_jobs=-1, cv=10)
if not sys.warnoptions:
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
grid_lr.fit(X, Y)
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score:.4f}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_:.4f}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')
```

### 5-fold cross validation (10 times)

```{code-cell}
name = "NN"
AUCs = list()
bp = grid_lr.best_params_
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
clf = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

count = 0
for train_index, test_index in rkf.split(X):
    count = count + 1

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred)
    AUCs.append(auc)

    X_header = np.array(X_train.columns)
    data_array = np.vstack((X_header, clf.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{count}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

# Fit predictor to statistically significant features (just once!!!)
clf.fit(X, Y)
y_pred = clf.predict_proba(X)[:,1]

# This in-sample AUC should be better than the AUCs from the repeated cross-validation
auc = roc_auc_score(Y, y_pred)

#AUC results from the 50 predictors
AUC_out = pd.DataFrame(AUCs, columns=['AUC'])
AUC_out.to_csv(f"data/{name}_AUCs.txt", sep='\t',index=False)

AUC_std= st.stdev(AUCs)
AUC_mean= st.mean(AUCs)

print(f'In-Sample AUC: {auc:.4f}')
print(f'MeanCV AUC: {AUC_mean:.4f}')
print(f'Standard Deviation CV AUC: {AUC_std:.4f}')
```


## NI

### Logistic regression model

```{code-cell}
table1 = table[table.inv_genotype=="NI"]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()
```

### Grid search for 3 hyperparameters

```{code-cell}
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
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score:.4f}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_:.4f}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')
```


### 5-fold cross validation (10 times)

```{code-cell}
name = "NI"
AUCs = list()
bp = grid_lr.best_params_
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
clf = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

count = 0
for train_index, test_index in rkf.split(X):
    count = count + 1

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred)
    AUCs.append(auc)

    X_header = np.array(X_train.columns)
    data_array = np.vstack((X_header, clf.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{count}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

# Fit predictor to statistically significant features (just once!!!)
clf.fit(X, Y)
y_pred = clf.predict_proba(X)[:,1]

# This in-sample AUC should be better than the AUCs from the repeated cross-validation
auc = roc_auc_score(Y, y_pred)

#AUC results from the 50 predictors
AUC_out = pd.DataFrame(AUCs, columns=['AUC'])
AUC_out.to_csv(f"data/{name}_AUCs.txt", sep='\t',index=False)

AUC_std= st.stdev(AUCs)
AUC_mean= st.mean(AUCs)

print(f'In-Sample AUC: {auc:.4f}')
print(f'MeanCV AUC: {AUC_mean:.4f}')
print(f'Standard Deviation CV AUC: {AUC_std:.4f}')
```


## II

### Logistic regression model

```{code-cell}
table1 = table[table.inv_genotype=="II"]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()
```

### Grid search for 3 hyperparameters

```{code-cell}
# parameters = {'C': [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 10, 20, 30],
#               'max_iter': [10, 25, 50, 75, 100, 150, 200, 400, 800, 1600],
#               'l1_ratio': [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}

parameters = {'C': [0.005, 0.01, 0.02],
              'max_iter': [50, 100, 200],
              'l1_ratio': [0.2, 0.1]}

grid_lr = GridSearchCV(lr, parameters, verbose=False, scoring='roc_auc', n_jobs=-1, cv=10)
if not sys.warnoptions:
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
grid_lr.fit(X, Y)
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score:.4f}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_:.4f}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')
```

### 5-fold cross validation (10 times)

```{code-cell}
name = "II"
AUCs = list()
bp = grid_lr.best_params_
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
clf = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

count = 0
for train_index, test_index in rkf.split(X):
    count = count + 1

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred)
    AUCs.append(auc)

    X_header = np.array(X_train.columns)
    data_array = np.vstack((X_header, clf.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{count}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

# Fit predictor to statistically significant features (just once!!!)
clf.fit(X, Y)
y_pred = clf.predict_proba(X)[:,1]

# This in-sample AUC should be better than the AUCs from the repeated cross-validation
auc = roc_auc_score(Y, y_pred)

#AUC results from the 50 predictors
AUC_out = pd.DataFrame(AUCs, columns=['AUC'])
AUC_out.to_csv(f"data/{name}_AUCs.txt", sep='\t',index=False)

AUC_std= st.stdev(AUCs)
AUC_mean= st.mean(AUCs)

print(f'In-Sample AUC: {auc:.4f}')
print(f'MeanCV AUC: {AUC_mean:.4f}')
print(f'Standard Deviation CV AUC: {AUC_std:.4f}')
```


## NN Males

### Logistic regression model

```{code-cell}
table1 = table[(table.gender == "M") & (table.inv_genotype=="NN")]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()
```

### Grid search for 3 hyperparameters

```{code-cell}
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
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score:.4f}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_:.4f}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')
```

### 5-fold cross validation (10 times)

```{code-cell}
name = "M-NN"
AUCs = list()
bp = grid_lr.best_params_
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
clf = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

count = 0
for train_index, test_index in rkf.split(X):
    count = count + 1

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred)
    AUCs.append(auc)

    X_header = np.array(X_train.columns)
    data_array = np.vstack((X_header, clf.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{count}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

# Fit predictor to statistically significant features (just once!!!)
clf.fit(X, Y)
y_pred = clf.predict_proba(X)[:,1]

# This in-sample AUC should be better than the AUCs from the repeated cross-validation
auc = roc_auc_score(Y, y_pred)

#AUC results from the 50 predictors
AUC_out = pd.DataFrame(AUCs, columns=['AUC'])
AUC_out.to_csv(f"data/{name}_AUCs.txt", sep='\t',index=False)

AUC_std= st.stdev(AUCs)
AUC_mean= st.mean(AUCs)

print(f'In-Sample AUC: {auc:.4f}')
print(f'MeanCV AUC: {AUC_mean:.4f}')
print(f'Standard Deviation CV AUC: {AUC_std:.4f}')
```


## NI Males

### Logistic regression model

```{code-cell}
table1 = table[(table.gender == "M") & (table.inv_genotype=="NI")]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()
```

### Grid search for 3 hyperparameters

```{code-cell}
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
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score:.4f}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_:.4f}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')
```

### 5-fold cross validation (10 times)

```{code-cell}
name = "M-NI"
AUCs = list()
bp = grid_lr.best_params_
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
clf = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

count = 0
for train_index, test_index in rkf.split(X):
    count = count + 1

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred)
    AUCs.append(auc)

    X_header = np.array(X_train.columns)
    data_array = np.vstack((X_header, clf.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{count}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

# Fit predictor to statistically significant features (just once!!!)
clf.fit(X, Y)
y_pred = clf.predict_proba(X)[:,1]

# This in-sample AUC should be better than the AUCs from the repeated cross-validation
auc = roc_auc_score(Y, y_pred)

#AUC results from the 50 predictors
AUC_out = pd.DataFrame(AUCs, columns=['AUC'])
AUC_out.to_csv(f"data/{name}_AUCs.txt", sep='\t',index=False)

AUC_std= st.stdev(AUCs)
AUC_mean= st.mean(AUCs)

print(f'In-Sample AUC: {auc:.4f}')
print(f'MeanCV AUC: {AUC_mean:.4f}')
print(f'Standard Deviation CV AUC: {AUC_std:.4f}')
```


## II Males

### Logistic regression model

```{code-cell}
table1 = table[(table.gender == "M") & (table.inv_genotype=="II")]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()
```

### Grid search for 3 hyperparameters

```{code-cell}
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
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score:.4f}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_:.4f}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')
```

### 5-fold cross validation (10 times)

```{code-cell}
name = "M-II"
AUCs = list()
bp = grid_lr.best_params_
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
clf = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

count = 0
for train_index, test_index in rkf.split(X):
    count = count + 1

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred)
    AUCs.append(auc)

    X_header = np.array(X_train.columns)
    data_array = np.vstack((X_header, clf.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{count}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

# Fit predictor to statistically significant features (just once!!!)
clf.fit(X, Y)
y_pred = clf.predict_proba(X)[:,1]

# This in-sample AUC should be better than the AUCs from the repeated cross-validation
auc = roc_auc_score(Y, y_pred)

#AUC results from the 50 predictors
AUC_out = pd.DataFrame(AUCs, columns=['AUC'])
AUC_out.to_csv(f"data/{name}_AUCs.txt", sep='\t',index=False)

AUC_std= st.stdev(AUCs)
AUC_mean= st.mean(AUCs)

print(f'In-Sample AUC: {auc:.4f}')
print(f'MeanCV AUC: {AUC_mean:.4f}')
print(f'Standard Deviation CV AUC: {AUC_std:.4f}')
```


## NN Females

### Logistic regression model

```{code-cell}
table1 = table[(table.gender == "F") & (table.inv_genotype=="NN")]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()
```

### Grid search for 3 hyperparameters

```{code-cell}
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
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score:.4f}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_:.4f}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')
```

### 5-fold cross validation (10 times)

```{code-cell}
name = "F-NN"
AUCs = list()
bp = grid_lr.best_params_
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
clf = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

count = 0
for train_index, test_index in rkf.split(X):
    count = count + 1

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred)
    AUCs.append(auc)

    X_header = np.array(X_train.columns)
    data_array = np.vstack((X_header, clf.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{count}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

# Fit predictor to statistically significant features (just once!!!)
clf.fit(X, Y)
y_pred = clf.predict_proba(X)[:,1]

# This in-sample AUC should be better than the AUCs from the repeated cross-validation
auc = roc_auc_score(Y, y_pred)

#AUC results from the 50 predictors
AUC_out = pd.DataFrame(AUCs, columns=['AUC'])
AUC_out.to_csv(f"data/{name}_AUCs.txt", sep='\t',index=False)

AUC_std= st.stdev(AUCs)
AUC_mean= st.mean(AUCs)

print(f'In-Sample AUC: {auc:.4f}')
print(f'MeanCV AUC: {AUC_mean:.4f}')
print(f'Standard Deviation CV AUC: {AUC_std:.4f}')
```


## NI Females

### Logistic regression model

```{code-cell}
table1 = table[(table.gender == "F") & (table.inv_genotype=="NI")]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()
```

### Grid search for 3 hyperparameters

```{code-cell}
# parameters = {'C': [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 10, 20, 30],
#              'max_iter': [10, 25, 50, 75, 100, 150, 200, 400, 800, 1600],
#              'l1_ratio': [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}

parameters = {'C': [0.005, 0.01],
              'max_iter': [10, 25],
              'l1_ratio': [1, 0.9]}

grid_lr = GridSearchCV(lr, parameters, verbose=False, scoring='roc_auc', n_jobs=-1, cv=10)
if not sys.warnoptions:
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
grid_lr.fit(X, Y)
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score:.4f}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_:.4f}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')
```

### 5-fold cross validation (10 times)

```{code-cell}
name = "F-NI"
AUCs = list()
bp = grid_lr.best_params_
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
clf = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

count = 0
for train_index, test_index in rkf.split(X):
    count = count + 1

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred)
    AUCs.append(auc)

    X_header = np.array(X_train.columns)
    data_array = np.vstack((X_header, clf.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{count}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

# Fit predictor to statistically significant features (just once!!!)
clf.fit(X, Y)
y_pred = clf.predict_proba(X)[:,1]

# This in-sample AUC should be better than the AUCs from the repeated cross-validation
auc = roc_auc_score(Y, y_pred)

#AUC results from the 50 predictors
AUC_out = pd.DataFrame(AUCs, columns=['AUC'])
AUC_out.to_csv(f"data/{name}_AUCs.txt", sep='\t',index=False)

AUC_std= st.stdev(AUCs)
AUC_mean= st.mean(AUCs)

print(f'In-Sample AUC: {auc:.4f}')
print(f'MeanCV AUC: {AUC_mean:.4f}')
print(f'Standard Deviation CV AUC: {AUC_std:.4f}')
```


## II Females

### Logistic regression model

```{code-cell}
table1 = table[(table.gender == "F") & (table.inv_genotype=="II")]
X = table1[table1.columns[5:]]
Y = table1['phenotype']
lr = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, penalty='elasticnet')
table1.groupby('phenotype')['participant_id'].nunique()
```

### Grid search for 3 hyperparameters

```{code-cell}
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
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(Y, best_lr.predict_proba(X)[:, 1])
coefs = best_lr.coef_[0, :]
num_coef = np.sum(coefs != 0)
X_header = np.array(X.columns)

data_array = np.vstack((X_header, coefs))
model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
print(f'Max AUC score:{max_auc_score:.4f}\n')
print(f'Non-zero coefficients: {num_coef}\n')
print(f'Best estimator: {grid_lr.best_estimator_}')
print(f'Scorer: {grid_lr.scorer_}')
print(f'Best params: {grid_lr.best_params_}')
print(f'Best score: {grid_lr.best_score_:.4f}\n')
m = model_coefs[model_coefs['Coefficient'] != 0 ].sort_values(by='Coefficient')
m = m.reset_index(drop=True).assign(Index=range(len(m)))
m.Index= m.Index + 1
m.set_index('Index')
```

### 5-fold cross validation (10 times)

```{code-cell}
name = "F-II"
AUCs = list()
bp = grid_lr.best_params_
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
clf = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

count = 0
for train_index, test_index in rkf.split(X):
    count = count + 1

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred)
    AUCs.append(auc)

    X_header = np.array(X_train.columns)
    data_array = np.vstack((X_header, clf.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{count}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

# Fit predictor to statistically significant features (just once!!!)
clf.fit(X, Y)
y_pred = clf.predict_proba(X)[:,1]

# This in-sample AUC should be better than the AUCs from the repeated cross-validation
auc = roc_auc_score(Y, y_pred)

#AUC results from the 50 predictors
AUC_out = pd.DataFrame(AUCs, columns=['AUC'])
AUC_out.to_csv(f"data/{name}_AUCs.txt", sep='\t',index=False)

AUC_std= st.stdev(AUCs)
AUC_mean= st.mean(AUCs)

print(f'In-Sample AUC: {auc:.4f}')
print(f'MeanCV AUC: {AUC_mean:.4f}')
print(f'Standard Deviation CV AUC: {AUC_std:.4f}')
```
