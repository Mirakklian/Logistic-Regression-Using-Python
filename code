import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#% matplotlib inline

data=pd.read_csv("C:\\Users\\Pratik Dutta\\Desktop\\DATASET.csv")
print(data.head(10))
print("# no of passenger in the data set:", +(len(data)))

## Data Wrangling----remove the unwanted dataset
print(data.isnull().sum())
sns.heatmap(data.isnull(),yticklabels=False,cmap="viridis")
plt.show()

pol=pd.get_dummies(data['Polarity'],drop_first=True) #drop the first colum..if positive=0,neutral=0..then it negative
print(pol)

##Concatinate the data field into our data set

data=pd.concat([data,pol],axis=1)
data.drop(['Polarity','tweet'],axis=1,inplace=True)
print(data.head(10))

## TRAIN MY DATASET

x=data.drop("positive",axis=1)
y=data["positive"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
print(logmodel.fit(x_train,y_train))

predictions=logmodel.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))  #generate classification report

##generate  accrucy

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))

from sklearn.metrics import accuracy_score

print( "The Accuracy of the prediction using Logistic Regression is: ",accuracy_score(y_test,predictions))

