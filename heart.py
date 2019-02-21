#%% [markdown]
# # Heart Disease UCI KAGGLE

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

#%% [markdown]
df=pd.read_csv("/home/rahul/Desktop/Link to rahul_environment/Projects/Machine_Learning Projects/Heart Disease/heart.csv")
df.head()
df.target.value_counts()
df.describe()
df.info()

#%% [markdown]
sns.countplot(x='target',data=df,palette='bwr')
plt.savefig("countplottarget")
plt.show()

#%% [markdown]
countNodisease=len(df['target']==0)
countHavedisease=len(df['target']==1)
print("Percentage of people don't have the  disease is",format(countNodisease/len(df.target)))
print("Percentage of people have the heart disease is",format(countNodisease/len(df.target)))

#%% [markdown]

sns.countplot(x='sex',data=df)
plt.savefig("countplot_sex")
plt.show()

#%% [markdown]
sns.countplot(x='sex',data=df,hue='target')
plt.savefig("countplot_sex_vs_target")
plt.show()

#%% [markdown]
countMale=len(df[df.sex==0])
countFemale=len(df[df.sex==1])
print("The no of the female having the disease",countFemale)
print("The no of male having the disease is",countMale)
df.groupby('target').mean()


#%% [markdown]
# # Checking about various attributes
male =len(df[df['sex'] == 1])
female = len(df[df['sex']== 0])
labels='Male','Female'
sizes=[male,female]
plt.pie(sizes,labels=labels)
plt.savefig('pie_male_vs_female')
plt.show()


#%% [markdown]
# # Various types of Chest Pain Types:
labels='Chest Pain Type 0','Chest Pain Type 1','Chest Pain Type 2','Chest Pain Type 3'
sizes=[len(df[df['cp'] == 0]),len(df[df['cp'] == 1]),len(df[df['cp'] ==2]),len(df[df['cp'] == 3])]
plt.pie(sizes,labels=labels)
plt.axis('equal')
plt.savefig("chestpain_types")
plt.show()

# ## We can make the more plots with fbs, thalach etc and find the interesting result according to it
# ## EXploratory Data Analysis

#%% [markdown]
plt.figure(figsize=(18,20))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.savefig("heatmap")
plt.show()

#%% [markdown]
sns.distplot(df['thalach'],kde=False,color='purple')
plt.savefig("thalach_in_distplot")
plt.show()


#%% [markdown]
sns.distplot(df['chol'],kde=False,color='purple')
plt.savefig("cholestrol_in_distplot")
plt.show()

#%% [markdown]
sns.distplot(df['fbs'],kde=False,color='purple')
plt.savefig("fbs_in_distplot")
plt.show()


#%% [markdown]
sns.countplot(x='age',data=df,hue='target')
plt.savefig("age_in_countplot")
plt.show()


#%% [markdown]
sns.scatterplot(x='chol',y='fbs',data=df,hue='target')
plt.savefig("cholestorol_in_scatterplot")
plt.show()

#%% [markdown]
sns.scatterplot(x='age',y='trestbps',data=df,hue='target')
plt.savefig("age_in_scatterplot")
plt.show()