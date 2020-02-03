---
title: 'Automatic Removal of Low-Correlating Features'
date: 2020-02-02
permalink: /posts/2020/02/Automatic-removal-1/
tags:
  - Machine Learning
  - Feature Engineering
  - Python
---
  
  this is a sample blog post.
kk

{% highlight python %}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

c = train.corr()
cv = c.values; length = cv.shape[0]-1;
cv_last = cv[length,:]
low_corr = np.argwhere(cv_last < 0.5)
train2 = train
low_corr=np.transpose(low_corr)

n_cols = len(train.columns)
num_cols = np.zeros((1, n_cols))
k = 0
for i in range(0,n_cols):
    if type(train.iloc[0][train.columns[i]])!=str:
        num_cols[0,k]=i; k = k + 1;
np.delete(num_cols,range(k,n_cols))
to_delete=num_cols[0,low_corr]

train2 = train.drop(train.columns[to_delete.astype(int)],  axis='columns')
plt.figure(figsize=[30,15])
sns.heatmap(train2.corr(), square=True, annot=True, annot_kws={"fontsize":18}, fmt=".2f");

{% endhighlight %}
