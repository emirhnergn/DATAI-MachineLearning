#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#%%

df = pd.read_csv("random-forest-regression-dataset.csv",sep=";",header=None)
print(df.head())

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%%

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100,random_state = 42)

rf.fit(x,y)

print("7.8 seviyesinde ki fiyat:",rf.predict([[7.8]]))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x)
y_head_ = rf.predict(x_)

#%%

plt.scatter(x,y,color="red")
plt.plot(x_,y_head_,color="green")
plt.xlabel("Tribun Level")
plt.ylabel("Price")
plt.show()


#%%

from sklearn.metrics import r2_score

print("RandomForest R2_Score:",r2_score(y,y_head))


#%%

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x,y)
y_head_lr = lr.predict(x)

print("Linear R2_Score:",r2_score(y,y_head_lr))

#%%

































#%%
#%%