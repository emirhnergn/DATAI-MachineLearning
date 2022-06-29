#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%

df = pd.read_csv("polynomial-regression.csv",sep=";")
print(df.head())


#%%

x = df.araba_fiyat.values.reshape(-1,1)
y = df.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("araba_fiyat")
plt.ylabel("araba_max_hiz")
plt.show()

#%%

# linear regression: y = b0+b1*x
# multiple linear regression: y = b0 + b1*x1 + b2*x2 + bn-xn

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

#%%

y_head = lr.predict(x)
plt.scatter(x,y)
plt.plot(x,y_head,color="red")
plt.show()


#%%

print(lr.predict([[10000]]))


#%%

# polynomial regression: y = b0+b1*x + b2*x^2 + b3*x^3 + bn*x^n
from sklearn.preprocessing import PolynomialFeatures

poly_degree = PolynomialFeatures(degree=2)
x_polynomial = poly_degree.fit_transform(x)

poly_degree4 = PolynomialFeatures(degree=4)
x_polynomial4 = poly_degree4.fit_transform(x)

#%%
poly_linear_regres = LinearRegression()
poly_linear_regres.fit(x_polynomial,y)

poly_linear_regres4 = LinearRegression()
poly_linear_regres4.fit(x_polynomial4,y)

#%%

y_head2 = poly_linear_regres.predict(x_polynomial)
y_head4 = poly_linear_regres4.predict(x_polynomial4)

plt.plot(x,y_head4,color="black",label="poly degree = 4")
plt.plot(x,y_head2,color="green",label="poly degree = 2")
plt.plot(x,y_head,color="red",label="linear")
plt.scatter(x,y,color="blue")
plt.legend()
plt.show()





#%%



































































#%%
#%%
#%%