#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

df = pd.read_csv("decision-tree-regression-dataset.csv",sep=";",header=None)
print(df.head())



#%%

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%%

from sklearn.tree import DecisionTreeRegressor

#%%

tree_reg = DecisionTreeRegressor(random_state=0)
tree_reg.fit(x,y)

y_head = tree_reg.predict(x)

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head_ = tree_reg.predict(x_)
#%%

plt.scatter(x,y,color="red")
plt.plot(x,y_head,color="green")
plt.plot(x_,y_head_,color="blue")
plt.xlabel("Tribun Level")
plt.ylabel("Price")
plt.show()

















#%%








































#%%
#%%