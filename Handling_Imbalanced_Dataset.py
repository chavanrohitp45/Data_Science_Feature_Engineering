# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:38:25 2024

@author: Rohit Chavan
"""

import numpy as np
import pandas as pd

# Set the random seed for reproducibility
np.random.seed(123)

# Create a dataframe with two classes
n_samples = 1000
class_0_ratio = 0.9
n_class_0 = int(n_samples * class_0_ratio) #900
n_class_1 = n_samples - n_class_0 # 100

# Create a dataframe with imbalanced dataset
class_0 = pd.DataFrame({
        'feature1' : np.random.normal(loc=0, scale=1, size=n_class_0),
        'feature2' : np.random.normal(loc=0, scale=1, size=n_class_0),
        'target':[0]*n_class_0
    })

class_1 = pd.DataFrame({
        'feature1' : np.random.normal(loc=2, scale=1, size=n_class_1),
        'feature2' : np.random.normal(loc=2, scale=1, size=n_class_1),
        'target':[1]*n_class_1
    })

df = pd.concat([class_0, class_1]).reset_index(drop=True)
df.head()
df['target'].value_counts()

df_minority = df[df['target'] == 1]
df_majority = df[df['target'] == 0]

# Upsampling
from sklearn.utils import resample
df_minority_upsampled = resample(df_minority, replace=True,n_samples=len(df_majority), random_state=42)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled['target'].value_counts()

##############################################

n_samples = 1000
class_0_ratio = 0.9
n_class_0 = int(n_samples * class_0_ratio) #900
n_class_1 = n_samples - n_class_0 # 100

# Create a dataframe with imbalanced dataset
class_0 = pd.DataFrame({
        'feature1' : np.random.normal(loc=0, scale=1, size=n_class_0),
        'feature2' : np.random.normal(loc=0, scale=1, size=n_class_0),
        'target':[0]*n_class_0
    })

class_1 = pd.DataFrame({
        'feature1' : np.random.normal(loc=2, scale=1, size=n_class_1),
        'feature2' : np.random.normal(loc=2, scale=1, size=n_class_1),
        'target':[1]*n_class_1
    })

df = pd.concat([class_0, class_1]).reset_index(drop=True)
df.head()
df['target'].xzzvalue_counts()

df_minority = df[df['target'] == 1]
df_majority = df[df['target'] == 0]

# Downsampling
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
df_downsampled = pd.concat([df_minority, df_majority_downsampled])
df_downsampled['target'].value_counts()


##############################################

# Handeling imbalanced dataset using SMOTE
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000,n_redundant=0, n_features=2, n_clusters_per_class=1, weights=[0.90], random_state=12)
import pandas as pd
df1 = pd.DataFrame(X, columns=['f1', 'f2'])
df2 = pd.DataFrame(y, columns=['target'])
final_df = pd.concat([df1, df2], axis=1)
final_df.head()

final_df['target'].value_counts()

import matplotlib.pyplot as plt
plt.scatter(final_df['f1'], final_df['f2'], c = final_df['target'])

from imblearn.over_sampling import SMOTE
# Transform the dataset 
oversample = SMOTE()
X, y = oversample.fit_resample(final_df[['f1', 'f2']], final_df['target'])

len(y)
len(y[y==1])
len(y[y==0])

df1 = pd.DataFrame(X, columns=['f1', 'f2'])
df2 = pd.DataFrame(y, columns=['target'])
final_df = pd.concat([df1, df2], axis=1)
plt.scatter(final_df['f1'], final_df['f2'], c = final_df['target'])
