import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

data=pd.read_csv('IRIS.csv')
# print(data.species.unique())
# print(data.species.value_counts())
x=data[['sepal length','sepal width','petal length','petal width']]
y=data.species
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

model=DecisionTreeClassifier(criterion='entropy')
model.fit(x_train,y_train)
predi=model.predict(x_test)
# print(pd.Series(y_test).value_counts())
# print(pd.Series(predi).value_counts())
print(pd.crosstab(y_test,predi))
print(confusion_matrix(y_test,predi))
print(classification_report(y_test,predi))
print(np.mean(predi==y_test))