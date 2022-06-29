#%%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%

df = pd.read_csv("linear-regression-dataset.csv",sep=";")
print(df.head())

#%%
plt.scatter(df.maas,df.deneyim)
plt.show()

#%%

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#%%

linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)
#print(x)
#print(y)

#%%

linear_reg.fit(x,y)
b0 = linear_reg.predict([[0]])
print("b0:",b0)

b0_ = linear_reg.intercept_
print("b0_:",b0_) #Y eksenini kestiği nokta

b1 = linear_reg.coef_
print("b1:",b1) #Eğim

# maas = 1663 + 1138*deneyim

maas_yeni = 1663+1138*11
print(maas_yeni)

print(linear_reg.predict([[11]])) 



#%%

array = np.array([i for i in range(0,16)]).reshape(-1,1)

plt.scatter(x,y,color="green")

y_head = linear_reg.predict(array)

plt.plot(array,y_head,color="red")
plt.show()







#%%







































































#%%
#%%
#%%