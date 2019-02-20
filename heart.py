#%% [markdown]
# # College Predictor In Pytho3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import sklearn
import io
import requests
import seaborn as sns
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

df=pd.read_csv("/home/rahul/Desktop/Link to rahul_environment/Projects/Machine_Learning Projects/Heart Disease/heart.csv")
df.head()


df.target.value_counts()

sns.countplot(x='target',data=df,palette='bwr')
plt.savefig("countplottarget")
plt.show()


countNodisease=len(df['target']==0)
countHavedisease=len(df['target']==1)

print("Percentage of people don't have the  disease is",format(countNodisease/len(df.target)))
print("Percentage of people have the heart disease is",format(countNodisease/len(df.target)))

sns.countplot(x='sex',data=df)
plt.savefig("countplotsex")
plt.show()
sns.countplot(x='sex',data=df,hue='target')
plt.savefig("countplotsexvstarget")


countMale=len(df[df.sex==0])
countFemale=len(df[df.sex==1])
print("The no of the female having the disease",countFemale)
print("The no of male having the disease is",countMale)

df.groupby('target').mean()



pd.crosstab()





