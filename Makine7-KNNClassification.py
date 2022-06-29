#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")

#%%

df = pd.read_csv("data.csv")
df.drop(["id","Unnamed: 32"],axis=1,inplace=True)
M = df[df["diagnosis"]=="M"]
B = df[df["diagnosis"]=="B"]

#%%
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu")
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

#%%

df.diagnosis = [1 if each=="M" else 0 for each in df.diagnosis]
y = df.diagnosis.values
x_data = df.drop(["diagnosis"],axis=1)

#%%
from sklearn.preprocessing import StandardScaler,MinMaxScaler
sscaler = StandardScaler()
sscaler.fit(x_data)
x_sscaler = sscaler.transform(x_data)

mmscaler = MinMaxScaler()
mmscaler.fit(x_data)
x_mmscaler = mmscaler.transform(x_data)
#xxx = pd.DataFrame(x_mmscaler)

# GEREKSİZ
"""
for i in range(len(x_data.columns)):
    mmscaler2 = MinMaxScaler()
    x_data[x_data.columns[i]] = mmscaler2.fit_transform(x_data[x_data.columns[i]].values.reshape(-1,1))
    del mmscaler2

print(x_data[x_data.columns[1]])"""

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)

#%%

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)
prediction_knn = knn.predict(x_test)
print("KNN Accuracy:",accuracy_score(y_test,prediction_knn))


#%%
"""
from sklearn.model_selection import GridSearchCV

grs = GridSearchCV(knn,{"n_neighbors":[i for i in range(1,100)]})
grs.fit(x_train,y_train)
print(grs.best_params_)
print(grs.best_score_)"""
#%%

score_list = []
for n in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors=n)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()

#%%

from sklearn.svm import SVC

svc1 = SVC(kernel="rbf",random_state=0)
svc1.fit(x_train,y_train)
predict_svc1 = svc1.predict(x_test)
print("SVC Score:",accuracy_score(y_test,predict_svc1))

#%%

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train,y_train)
print("GNB Score:",gnb.score(x_test,y_test))

#%%

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=0,criterion="gini")
dtc.fit(x_train,y_train)
print("DecisionTree Score:",dtc.score(x_test,y_test))

#%%

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=15,criterion="gini",random_state=0)
rfc.fit(x_train,y_train)
print("RandomForest Score:",rfc.score(x_test,y_test))

#%%

score_list = []
for n in range(1,30):
    rfc2 = RandomForestClassifier(n_estimators=n,criterion="gini",random_state=0)
    rfc2.fit(x_train,y_train)
    score_list.append(rfc2.score(x_test,y_test))

plt.plot(range(1,30),score_list)
plt.xlabel("n estimator values")
plt.ylabel("accuracy")
plt.show()

#%%















































#%%
#%%
#%%