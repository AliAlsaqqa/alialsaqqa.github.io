---
title: 'Automatic Removal of Low-Correlating Features'
date: 2020-02-02
permalink: /posts/2020/02/Automatic-removal-1/
tags:
  - Machine Learning
  - Feature Engineering
  - Python
---

Introduction
------
If you have been doing any Machine Learning recently, you have probably spent some time doing feature engineering. One important aspect of this task is to identify features that do not correlate well with the target.

Consider, for example, a typical regression problem where we want to predict the price of a house given (labeled) house data. This dataset will include many features, for instance the area of the house, number of rooms, location, etc, in addition to the predictor: sale price. It is usually a good practice to keep our models as simple as possible (but not any simpler). For that we have to identify which of the features given are important and which are not.

But how do we do that? One method is to look at the (Pearson) correlation coefficient between a given feature and the predictor (sale price). Those features that have low correlation (e.g. less than 0.5) are then dropped.

This task is usually done manually. **The goal of this post is to provide a simple Python script that automatically remove features with low correlation to the predictor.** We assume we are given a Pandas dataframe which have a mix of numeric and categorical columns, and that the predictor column is the last one among the numeric columns. Had the dataframe been purely numerical then few lines of code would have been enough to do the job (using `df.select_dtypes(include=['int'])`). But when there are categorical, as well as numerical, columns then that command will not work; it will still choose the numeric columns but there is no way to map these back (after dropping some of them that are deeped less important) to the originial dataframe.

Implementation
------
We start by showing a typical heatmap that illustrate the concept. The dataset is from a [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The dataframe contains a mix of columns. The following command generate the typical heatmap: `sns.heatmap(train.corr(), annot=True, fmt=".1f");`:

![Heatmap with all features included](/images/heat1.png)

The usual course of action is to look at the figure, identify features which correlate to the predictor by less than a threshold, and then manually drop them. Instead the following code does the job automatically. We assume the dataframe is called 'train', the output dataframe is then called 'train2':

{% highlight python %}
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Input data
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

c = train.corr() # Get correlation matrix
cv = c.values; length = cv.shape[0]-1; # convert  to numeric array
cv_last = cv[length,:] # Assume predictor is the last column
low_corr = np.argwhere(cv_last < 0.5) # Identify columns that have corr < 0.5
train2 = train
low_corr=np.transpose(low_corr)

n_cols = len(train.columns)
num_cols = np.zeros((1, n_cols))
k = 0
for i in range(0,n_cols): # Purpose of this loop is to get an index of the numeric columns within the original mixed dataframe
    if type(train.iloc[0][train.columns[i]])!=str:
        num_cols[0,k]=i; k = k + 1;
np.delete(num_cols,range(k,n_cols)) # Remove extra columns
to_delete=num_cols[0,low_corr] # index of columns to be deleted

train2 = train.drop(train.columns[to_delete.astype(int)],  axis='columns') # Drop the low correlating columns

plt.figure(figsize=[30,15])
sns.heatmap(train2.corr(), square=True, annot=True, annot_kws={"fontsize":18}, fmt=".2f");

{% endhighlight %}

Here is the heatmap with low-correlating features removed:

![Heatmap with some features included](/images/heat2.png)

Conclusion
==========
In this post we provided a short script to automate part of your feature engineering work. I hope you find the code usefull. IF you have any comment please leave it below!
