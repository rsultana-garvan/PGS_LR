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


# Cooper 142 SNPs set (10 splits)


## Preparation

### Import required packages.

```{code-cell}
import matplotlib.pyplot as plt
import os, sys, warnings
import numpy as np
import pandas as pd
import statistics as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, auc, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, RepeatedKFold, StratifiedKFold
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
y = table['phenotype']
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
grid_lr.fit(X, y)
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(y, best_lr.predict_proba(X)[:, 1])
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

### ROC with cross-validation

```{code-cell}
# Run classifier with cross-validation and plot ROC curves
name = "all"
bp = grid_lr.best_params_
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
classifier = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.rcParams['figure.figsize'] = [14, 12]
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X.iloc[train,], y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X.iloc[test,],
        y.iloc[test],
        name=f"ROC fold {i + 1}",
        alpha=0.2,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    X_header = np.array(X.columns)
    data_array = np.vstack((X_header, classifier.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{i+1:02}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

classifier.fit(X, y)
viz = RocCurveDisplay.from_estimator(
    classifier,
    X,
    y,
    name=f"In-sample ROC",
    alpha=0.6,
    color="k",
    lw=2,
    ax=ax,
)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.6,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.4,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title=f"Receiver operating characteristic for {name} samples",
)
ax.legend(loc="lower right", fontsize='xx-small'
)
plt.show()
```


## Males

### Logistic regression model

```{code-cell}
table1 = table[table.gender == "M"]
X = table1[table1.columns[5:]]
y = table1['phenotype']
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
grid_lr.fit(X, y)
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(y, best_lr.predict_proba(X)[:, 1])
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

### ROC with cross-validation

```{code-cell}
# Run classifier with cross-validation and plot ROC curves
name = "M"
bp = grid_lr.best_params_
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
classifier = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.rcParams['figure.figsize'] = [14, 12]
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X.iloc[train,], y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X.iloc[test,],
        y.iloc[test],
        name=f"ROC fold {i + 1}",
        alpha=0.2,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    X_header = np.array(X.columns)
    data_array = np.vstack((X_header, classifier.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{i+1:02}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

classifier.fit(X, y)
viz = RocCurveDisplay.from_estimator(
    classifier,
    X,
    y,
    name=f"In-sample ROC",
    alpha=0.6,
    color="k",
    lw=2,
    ax=ax,
)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.6,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.4,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title=f"Receiver operating characteristic for {name} samples",
)
ax.legend(loc="lower right", fontsize='xx-small'
)
plt.show()
```


## Females

### Logistic regression model

```{code-cell}
table1 = table[table.gender == "F"]
X = table1[table1.columns[5:]]
y = table1['phenotype']
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
grid_lr.fit(X, y)
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(y, best_lr.predict_proba(X)[:, 1])
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

### ROC with cross-validation

```{code-cell}
# Run classifier with cross-validation and plot ROC curves
name = "F"
bp = grid_lr.best_params_
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
classifier = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.rcParams['figure.figsize'] = [14, 12]
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X.iloc[train,], y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X.iloc[test,],
        y.iloc[test],
        name=f"ROC fold {i + 1}",
        alpha=0.2,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    X_header = np.array(X.columns)
    data_array = np.vstack((X_header, classifier.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{i+1:02}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

classifier.fit(X, y)
viz = RocCurveDisplay.from_estimator(
    classifier,
    X,
    y,
    name=f"In-sample ROC",
    alpha=0.6,
    color="k",
    lw=2,
    ax=ax,
)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.6,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.4,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title=f"Receiver operating characteristic for {name} samples",
)
ax.legend(loc="lower right", fontsize='xx-small'
)
plt.show()
```


## NN

### Logistic regression model

```{code-cell}
table1 = table[table.inv_genotype=="NN"]
X = table1[table1.columns[5:]]
y = table1['phenotype']
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
grid_lr.fit(X, y)
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(y, best_lr.predict_proba(X)[:, 1])
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

### ROC with cross-validation

```{code-cell}
# Run classifier with cross-validation and plot ROC curves
name = "NN"
bp = grid_lr.best_params_
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
classifier = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.rcParams['figure.figsize'] = [14, 12]
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X.iloc[train,], y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X.iloc[test,],
        y.iloc[test],
        name=f"ROC fold {i + 1}",
        alpha=0.2,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    X_header = np.array(X.columns)
    data_array = np.vstack((X_header, classifier.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{i+1:02}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

classifier.fit(X, y)
viz = RocCurveDisplay.from_estimator(
    classifier,
    X,
    y,
    name=f"In-sample ROC",
    alpha=0.6,
    color="k",
    lw=2,
    ax=ax,
)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.6,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.4,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title=f"Receiver operating characteristic for {name} samples",
)
ax.legend(loc="lower right", fontsize='xx-small'
)
plt.show()
```


## NI

### Logistic regression model

```{code-cell}
table1 = table[table.inv_genotype=="NI"]
X = table1[table1.columns[5:]]
y = table1['phenotype']
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
grid_lr.fit(X, y)
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(y, best_lr.predict_proba(X)[:, 1])
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

### ROC with cross-validation

```{code-cell}
# Run classifier with cross-validation and plot ROC curves
name = "NI"
bp = grid_lr.best_params_
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
classifier = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.rcParams['figure.figsize'] = [14, 12]
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X.iloc[train,], y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X.iloc[test,],
        y.iloc[test],
        name=f"ROC fold {i + 1}",
        alpha=0.2,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    X_header = np.array(X.columns)
    data_array = np.vstack((X_header, classifier.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{i+1:02}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

classifier.fit(X, y)
viz = RocCurveDisplay.from_estimator(
    classifier,
    X,
    y,
    name=f"In-sample ROC",
    alpha=0.6,
    color="k",
    lw=2,
    ax=ax,
)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.6,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.4,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title=f"Receiver operating characteristic for {name} samples",
)
ax.legend(loc="lower right", fontsize='xx-small'
)
plt.show()
```


## II

### Logistic regression model

```{code-cell}
table1 = table[table.inv_genotype=="II"]
X = table1[table1.columns[5:]]
y = table1['phenotype']
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
grid_lr.fit(X, y)
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(y, best_lr.predict_proba(X)[:, 1])
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

### ROC with cross-validation

```{code-cell}
# Run classifier with cross-validation and plot ROC curves
name = "II"
bp = grid_lr.best_params_
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
classifier = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.rcParams['figure.figsize'] = [14, 12]
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X.iloc[train,], y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X.iloc[test,],
        y.iloc[test],
        name=f"ROC fold {i + 1}",
        alpha=0.2,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    X_header = np.array(X.columns)
    data_array = np.vstack((X_header, classifier.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{i+1:02}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

classifier.fit(X, y)
viz = RocCurveDisplay.from_estimator(
    classifier,
    X,
    y,
    name=f"In-sample ROC",
    alpha=0.6,
    color="k",
    lw=2,
    ax=ax,
)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.6,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.4,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title=f"Receiver operating characteristic for {name} samples",
)
ax.legend(loc="lower right", fontsize='xx-small'
)
plt.show()
```


## NN Males

### Logistic regression model

```{code-cell}
table1 = table[(table.gender == "M") & (table.inv_genotype=="NN")]
X = table1[table1.columns[5:]]
y = table1['phenotype']
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
grid_lr.fit(X, y)
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(y, best_lr.predict_proba(X)[:, 1])
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

### ROC with cross-validation

```{code-cell}
# Run classifier with cross-validation and plot ROC curves
name = "M-NN"
bp = grid_lr.best_params_
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
classifier = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.rcParams['figure.figsize'] = [14, 12]
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X.iloc[train,], y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X.iloc[test,],
        y.iloc[test],
        name=f"ROC fold {i + 1}",
        alpha=0.2,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    X_header = np.array(X.columns)
    data_array = np.vstack((X_header, classifier.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{i+1:02}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

classifier.fit(X, y)
viz = RocCurveDisplay.from_estimator(
    classifier,
    X,
    y,
    name=f"In-sample ROC",
    alpha=0.6,
    color="k",
    lw=2,
    ax=ax,
)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.6,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.4,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title=f"Receiver operating characteristic for {name} samples",
)
ax.legend(loc="lower right", fontsize='xx-small'
)
plt.show()
```


## NI Males

### Logistic regression model

```{code-cell}
table1 = table[(table.gender == "M") & (table.inv_genotype=="NI")]
X = table1[table1.columns[5:]]
y = table1['phenotype']
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
grid_lr.fit(X, y)
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(y, best_lr.predict_proba(X)[:, 1])
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

### ROC with cross-validation

```{code-cell}
# Run classifier with cross-validation and plot ROC curves
name = "M-NI"
bp = grid_lr.best_params_
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
classifier = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.rcParams['figure.figsize'] = [14, 12]
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X.iloc[train,], y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X.iloc[test,],
        y.iloc[test],
        name=f"ROC fold {i + 1}",
        alpha=0.2,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    X_header = np.array(X.columns)
    data_array = np.vstack((X_header, classifier.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{i+1:02}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

classifier.fit(X, y)
viz = RocCurveDisplay.from_estimator(
    classifier,
    X,
    y,
    name=f"In-sample ROC",
    alpha=0.6,
    color="k",
    lw=2,
    ax=ax,
)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.6,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.4,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title=f"Receiver operating characteristic for {name} samples",
)
ax.legend(loc="lower right", fontsize='xx-small'
)
plt.show()
```


## II Males

### Logistic regression model

```{code-cell}
table1 = table[(table.gender == "M") & (table.inv_genotype=="II")]
X = table1[table1.columns[5:]]
y = table1['phenotype']
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
grid_lr.fit(X, y)
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(y, best_lr.predict_proba(X)[:, 1])
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

### ROC with cross-validation

```{code-cell}
# Run classifier with cross-validation and plot ROC curves
name = "M-II"
bp = grid_lr.best_params_
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
classifier = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.rcParams['figure.figsize'] = [14, 12]
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X.iloc[train,], y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X.iloc[test,],
        y.iloc[test],
        name=f"ROC fold {i + 1}",
        alpha=0.2,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    X_header = np.array(X.columns)
    data_array = np.vstack((X_header, classifier.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{i+1:02}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

classifier.fit(X, y)
viz = RocCurveDisplay.from_estimator(
    classifier,
    X,
    y,
    name=f"In-sample ROC",
    alpha=0.6,
    color="k",
    lw=2,
    ax=ax,
)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.6,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.4,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title=f"Receiver operating characteristic for {name} samples",
)
ax.legend(loc="lower right", fontsize='xx-small'
)
plt.show()
```


## NN Females

### Logistic regression model

```{code-cell}
table1 = table[(table.gender == "F") & (table.inv_genotype=="NN")]
X = table1[table1.columns[5:]]
y = table1['phenotype']
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
grid_lr.fit(X, y)
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(y, best_lr.predict_proba(X)[:, 1])
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

### ROC with cross-validation

```{code-cell}
# Run classifier with cross-validation and plot ROC curves
name = "F-NN"
bp = grid_lr.best_params_
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
classifier = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.rcParams['figure.figsize'] = [14, 12]
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X.iloc[train,], y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X.iloc[test,],
        y.iloc[test],
        name=f"ROC fold {i + 1}",
        alpha=0.2,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    X_header = np.array(X.columns)
    data_array = np.vstack((X_header, classifier.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{i+1:02}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

classifier.fit(X, y)
viz = RocCurveDisplay.from_estimator(
    classifier,
    X,
    y,
    name=f"In-sample ROC",
    alpha=0.6,
    color="k",
    lw=2,
    ax=ax,
)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.6,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.4,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title=f"Receiver operating characteristic for {name} samples",
)
ax.legend(loc="lower right", fontsize='xx-small'
)
plt.show()
```


## NI Females

### Logistic regression model

```{code-cell}
table1 = table[(table.gender == "F") & (table.inv_genotype=="NI")]
X = table1[table1.columns[5:]]
y = table1['phenotype']
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
grid_lr.fit(X, y)
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(y, best_lr.predict_proba(X)[:, 1])
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

### ROC with cross-validation

```{code-cell}
# Run classifier with cross-validation and plot ROC curves
name = "F-NI"
bp = grid_lr.best_params_
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
classifier = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.rcParams['figure.figsize'] = [14, 12]
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X.iloc[train,], y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X.iloc[test,],
        y.iloc[test],
        name=f"ROC fold {i + 1}",
        alpha=0.2,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    X_header = np.array(X.columns)
    data_array = np.vstack((X_header, classifier.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{i+1:02}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

classifier.fit(X, y)
viz = RocCurveDisplay.from_estimator(
    classifier,
    X,
    y,
    name=f"In-sample ROC",
    alpha=0.6,
    color="k",
    lw=2,
    ax=ax,
)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.6,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.4,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title=f"Receiver operating characteristic for {name} samples",
)
ax.legend(loc="lower right", fontsize='xx-small'
)
plt.show()
```


## II Females

### Logistic regression model

```{code-cell}
table1 = table[(table.gender == "F") & (table.inv_genotype=="II")]
X = table1[table1.columns[5:]]
y = table1['phenotype']
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
grid_lr.fit(X, y)
```

### Best estimator

```{code-cell}
best_lr = grid_lr.best_estimator_

max_auc_score = roc_auc_score(y, best_lr.predict_proba(X)[:, 1])
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

### ROC with cross-validation

```{code-cell}
# Run classifier with cross-validation and plot ROC curves
name = "F-II"
bp = grid_lr.best_params_
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
classifier = LogisticRegression(C=bp['C'], max_iter=bp['max_iter'], l1_ratio=bp['l1_ratio'], random_state=42,
      solver='saga', n_jobs=-1, penalty='elasticnet')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.rcParams['figure.figsize'] = [14, 12]
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X.iloc[train,], y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X.iloc[test,],
        y.iloc[test],
        name=f"ROC fold {i + 1}",
        alpha=0.2,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    X_header = np.array(X.columns)
    data_array = np.vstack((X_header, classifier.coef_[0,:]))
    model_coefs = pd.DataFrame(data=data_array.T, columns=['SNP', 'Coefficient'])
    m_name = f'data/{name}_10fold_repeat{i+1:02}_coefficients.txt'
    model_coefs.to_csv(m_name, sep='\t',index=False)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

classifier.fit(X, y)
viz = RocCurveDisplay.from_estimator(
    classifier,
    X,
    y,
    name=f"In-sample ROC",
    alpha=0.6,
    color="k",
    lw=2,
    ax=ax,
)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.6,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.4,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title=f"Receiver operating characteristic for {name} samples",
)
ax.legend(loc="lower right", fontsize='xx-small'
)
plt.show()
```
