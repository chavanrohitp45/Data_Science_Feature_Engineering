# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:03:17 2024

@author: Rohit Chavan
"""

# One Hot Encoder

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({
        'color' : ['red', 'blue', 'green', 'green', 'red', 'blue']
    })

df.head()

# Create an instance of OneHotEncoder
encoder = OneHotEncoder()

# Perform fit and transform
encoded = encoder.fit_transform(df[['color']]).toarray()
encode_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
encode_df

pd.concat([df, encode_df], axis=1)

# Label Encoding : Alphabetical encoding

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

encoder.fit_transform(df[['color']])

# Ordinal Encoding : If we have a situation to ranking

from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({
        'size' : ['small', 'medium', 'large', 'large', 'medium', 'small']
    })

encoder = OrdinalEncoder(categories=[['small', 'medium', 'large']])

encoder.fit_transform(df[['size']])

# Target Guided Encoding
df = pd.DataFrame({
        'city': ['NewYork', 'London', 'Paris', 'Tokyo', 'NewYork', 'Paris'],
        'Price': [200, 150, 300, 250, 180, 320]
    })

df.groupby('city')['Price'].mean()
mean_price = df.groupby('city')['Price'].mean().to_dict()
df['city_encoded'] = df['city'].map(mean_price)
df[['city_encoded', 'Price']] # consider these two features to train our model

# Assignments based on Target Guided Encoding
import seaborn as sns
df = sns.load_dataset('tips')

df.head()
mean_val = df.groupby('time')['total_bill'].mean().to_dict()
df['time_encoded'] = df['time'].map(mean_val)
df[['time_encoded','total_bill']]
