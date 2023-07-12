#code
#Importing all the libraries



import pandas as pd

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#Importing the data
dataset = pd.read_csv('loan_data.csv',sep=',')

# print the length of data set,then printing the Data and finaly the Size.
print(len(dataset))
print('DataSet::',dataset.head())
print(dataset.shape)

#handling categorical variable purpose
'''
Here we are handling the second attribute PURPOSE as it is a string attribute so we need to remove 
it from the testing and train data as the numerical operation cannot be performed on String.

GET_DUMMIES --Convert categorical variable into dummy/indicator variables.
DROP_FIRST --Whether to get k-1 dummies out of k categorical levels by removing the first level
Axis=1  cloumn 
Axis=0 row (default)

'''
purpose_c = pd.get_dummies(dataset['purpose'], drop_first=True)
purpose_c
loans_f = pd.concat([dataset, purpose_c], axis=1).drop('purpose', axis=1)
loans_f.head()
print(loans_f.shape)

#Splitting the dataset into test and train set
y = loans_f['not.fully.paid'] 
X = loans_f.drop('not.fully.paid', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

#calculating the Entropy and the prediction

clf_entropy = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
clf_entropy.fit(X_train, y_train)
prediction = clf_entropy.predict(X_test)
print(prediction)

#checking performance of the model
print(confusion_matrix(y_test, prediction))
print(accuracy_score(y_test,prediction))

from matplotlib import pyplot as plt

text_representation = tree.export_text(clf_entropy)
print(text_representation)



fig = plt.figure(figsize=(70,50))
_ = tree.plot_tree(clf_entropy, filled=True)
fig.savefig("LoanPaymt.png")


# fror logistic regression
#IMPORTING THE IMPORTANT LIBRARIES 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#LOADING THE CSV FILE OF THE DATASET
file=pd.read_csv("filter.csv")
file.head(100)

#DEFINING THE INDEPENDENT AND DEPENDENT VARIABLE 
x = file["article_content"]
y = file["labels"]

#SPLITTING THE TRAINING AND TESTING 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

#CONVERT THE text into vectors

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)

pred_lr=LR.predict(xv_test)
pre=LR.score(xv_test, y_test)
print(pre)
print(classification_report(y_test, pred_lr))

