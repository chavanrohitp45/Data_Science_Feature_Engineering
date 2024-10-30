# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:09:41 2024

@author: Rohit Chavan
"""

import seaborn as sns

# titanic dataset is loaded
df = sns.load_dataset('titanic')

df.head()

# Check missing values
df.isnull()
df.isnull().sum()

# Delete the rows and data points to handel the missing values
df.shape
df.dropna().shape

# Delete Column 
df.dropna(axis=1).shape

# Imputation Techniques
# 1. Mean Value imputation : Normally distributed data
sns.histplot(df['age'], kde=True)
df['age'].fillna(df['age'].mean()).isnull().sum()

# 2. Median Value imputation : Right skewed and left skewed data / Outliers
df['age'].fillna(df['age'].median()).isnull().sum()

# 3. Mode value imputation : Categorical data
df['embarked'].isnull().sum()
df['embarked'].unique()
mode_val = df['embarked'].mode()[0]
df['embarked'].fillna(mode_val).isnull().sum()
