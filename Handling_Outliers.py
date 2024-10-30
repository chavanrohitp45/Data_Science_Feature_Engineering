# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:41:24 2024

@author: Rohit Chavan
"""

import numpy as np
lst_marks = [45,32,56,75,89,54,32,89,90,87,67,54,45,98,99,67,74]
minimum, Q1, median, Q3, maximum = np.quantile(lst_marks, [0, 0.25, 0.50, 0.75, 1.0])
minimum, Q1, median, Q3, maximum

IQR = Q3 - Q1

# Find lower fence
lower_fence = Q1 - 1.5*IQR
lower_fence

# Find higher fence
higher_fence = Q3 + 1.5*IQR
higher_fence

# Draw the boxplot
import seaborn as sns
sns.boxplot(lst_marks)

# now add outliers
lst_marks = [-100,-200,45,32,56,75,89,54,32,89,90,87,67,54,45,98,99,67,74,150,170,180]
sns.boxplot(lst_marks)
