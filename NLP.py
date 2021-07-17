# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:23:10 2020

@author: vipvi
"""

import numpy as np
import pandas as pd
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
import re
import joblib
import nltk #natural language toolkit
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
c=[]
for i in range(0,1000):
    
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) #Replacing text with space
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' ' .join(review)
    c.append(review)
    #building sparse matrix
from  sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(c).toarray() 
joblib.dump(cv.vocabulary_,"features.save")
y=dataset.iloc[:,-1].values
#Training and testing NLP
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#ANN, features are inputs
model=Sequential()
model.add(Dense(input_dim=1500,kernel_initializer='random_uniform',activation='sigmoid',units=1000))
model.add(Dense(units=100,kernel_initializer='random_uniform',activation='sigmoid'))
model.add(Dense(units=1,kernel_initializer='random_uniform',activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=50,batch_size=10)
y_pred=model.predict(x_test)
y_pred=(y_pred>=0.5)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
loaded=CountVectorizer(decode_error='replace',vocabulary=joblib.load('features.save'))
da="The service was not up to par, either."
da = da.split("delimiter")
result=model.predict(loaded.transform(da))
prediction=result>=0.5
print(prediction)

