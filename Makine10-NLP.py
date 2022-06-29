#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

data = pd.read_csv(r"gender-classifier.csv",encoding="latin1")
data = pd.concat([data.gender,data.description],axis=1)
data.dropna(inplace=True,axis=0)

data.gender = [ 1 if each == "female" else 0 for each in data.gender]

#%%

import re

first_description = data.description[4]
description = re.sub("[^a-zA-Z]"," ",first_description)
description = description.lower()

#%%

import nltk
#nltk.download('wordnet')
#nltk.download("stopwords")
#nltk.download("punkt")

#%%

from nltk.corpus import stopwords


#description = description.split()

#split yerine tokenizer kullanılabilir

description = nltk.word_tokenize(description)


#%%

description = [word for word in description if not word in set(stopwords.words("english"))]


#%%

from nltk.stem.porter import PorterStemmer
from nltk import WordNetLemmatizer

lemma = WordNetLemmatizer()
ps = PorterStemmer()

description1 = [lemma.lemmatize(word) for word in description]
#description2 = [ps.stem(word) for word in description]

#%%
del description
description = str()
lemma = WordNetLemmatizer()
description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    #description = [word for word in description if not word in set(stopwords.words("english"))]
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)
del description
    
#%%

from sklearn.feature_extraction.text import CountVectorizer

max_features = 5000

cv = CountVectorizer(max_features=max_features,stop_words="english") #lowercase = True,token_pattern = 

sparce_matrix = cv.fit_transform(description_list).toarray() 

#%%

#print("en sık kullanılan {} kelimeler:{}".format(max_features,cv.get_feature_names()))

#%%

y = data.iloc[:,0].values
x = sparce_matrix

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42)


#%%

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)

y_pred = nb.predict(x_test)
print("NB Accuracy:",nb.score(y_test.reshape(-1,1),y_pred.reshape(-1,1)))

#%%















#%%
#%%
#%%