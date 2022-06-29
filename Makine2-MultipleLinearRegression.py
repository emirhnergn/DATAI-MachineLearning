#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%

df = pd.read_csv("multiple-linear-regression-dataset.csv",sep=";")
print(df.head())

#%%

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

#%%

x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)
print(x)
print(y)

#%%

multiple_lr = LinearRegression()
multiple_lr.fit(x,y)

print("b0:",multiple_lr.intercept_)
print("b1,b2:",multiple_lr.coef_)

#%%

pre1 = multiple_lr.predict(np.array([[10,35],[5,35]]))
print(pre1)


#%%

from statsmodels.api import OLS

olss = OLS(endog=y,exog=x).fit()
print(olss.summary())
#%%



























#%%






















































#%%
#%%
#%5