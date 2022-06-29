#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")

#%%

from sklearn.datasets import load_iris

iris = load_iris()

#%%

x = iris.data
y = iris.target

#%%

x1 = (x-np.min(x)/(np.max(x)-np.min(x)))

#%%

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)


#%%

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3,metric="minkowski")

#%%
#K fold CV K = 10
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator= knn,X= x_train,y = y_train,cv=10)
print("avarage accuracy:",np.mean(accuracies))
print("avarage std:",np.std(accuracies))

#%%
knn.fit(x_train,y_train)
print("test accuracy:",knn.score(x_test,y_test))


#%%

from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors":np.arange(1,50)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn,grid,scoring="accuracy",cv=10)
knn_cv.fit(x,y)
print("Best estimator:",knn_cv.best_estimator_)
print("Best params:",knn_cv.best_params_)
print("Best score:",knn_cv.best_score_)

#%%

from sklearn.linear_model import LogisticRegression

x = x[:100,:]
y = y[:100]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)


grid = {"C":np.logspace(-3,3,7),
        "penalty":["l1","l2"], #l1 = lasso l2 = ridge
        }

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,scoring="accuracy",cv=10)
logreg_cv.fit(x_train,y_train)
print("Best estimator:",logreg_cv.best_estimator_)
print("Best params:",logreg_cv.best_params_)
print("Best score:",logreg_cv.best_score_)
#%%

logreg2 = LogisticRegression(C=0.01,penalty="l2")
logreg2.fit(x_train,y_train)
print("Logreg2 Score:",logreg2.score(x_test,y_test))
#%%



































#%%