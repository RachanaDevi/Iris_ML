# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:43:24 2019

@author: Rachu
"""


   
'''
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica
'''

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''
 Create new csv file 
 2 parameters
* csv file name
* the file which has the data

create_csv("iris.csv", " iris.data")
'''
def create_csv_file(csv_name,real_file):
    iris_csv = open("iris.csv", "w")
    with open("iris.data", "r") as f:
        iris_csv.write(f.read())
    iris_csv.close()
    dataset = pd.read_csv(csv_name, delimiter = ',', header =None,dtype='str')
    return dataset


'''
 Convert columns from object to int
 2 parameters
* dataset
* col_no

create_csv("iris.csv", " iris.data")
'''
def str_to_int(dataset,col_no):
    class_values = [row[col_no] for row in dataset.values]
    unique = set(class_values)
    lookup = dict()
    for i,value in enumerate(unique):
        lookup[value]=i
    for row in range(len(dataset)):        
        dataset.loc[row,col_no] = lookup[dataset.iloc[row, col_no]]      
         
'''
 Convert columns from object to int
 3 parameters
* dataset
* no of columns

create_csv("iris.csv", " iris.data")
'''
def str_column_to_float(dataset,col_no):
        for col in list(dataset.columns.values):
            if(col==4):
                break
            dataset[col]=dataset[col].astype('float')
          
            
    
csv_name = "iris.csv"
real_file="iris.data"
dataset = create_csv_file(csv_name, real_file)
print('Loaded data file {} with {} rows and {} columns '.format(csv_name,len(dataset), len(dataset.columns) ))
str_column_to_float(dataset,len(dataset.columns)-1)
str_to_int(dataset,4)

X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values

#Splitting the dataset in Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y ,train_size = 0.4, random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



#Fitting KNN to the Training Set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(metric='minkowski', p =2,n_neighbors = 5)
classifier.fit(X_train,y_train.ravel())


#Predicting the results
y_pred = classifier.predict(X_test)

#Creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))
