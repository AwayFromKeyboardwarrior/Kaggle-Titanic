import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#import reprlib as re
sns.set()
train = pd.read_csv('train.csv')
#train1 = pd.read_csv('C:\\Users\\kimji\\Desktop\\20190109_Kaggle\\train1.csv')
#test = pd.read_csv('C:\\Users\\kimji\\Desktop\\20190109_Kaggle\\test.csv')
#head=train.head()
#print(head)
#print(train.head())
#print(train[train['Survived']==1])
#print(type(train[list('')]))
#print(train[list('true')])
#print(train1[train1['1']])
#print(list([['true','he']['ha','hu']]))


#Survived = train[train['Survived'] == 1]['Sex'].value_counts()
#print(Survived)
#print(type(Survived))

def barchk(name):
    survived = train[train['Survived']==1][name].value_counts()
    dead = train[train['Survived']==0][name].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index=['survived','dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))
    plt.show()
    #print(df.filter(like='Mr.',axis=1))
    #print(df)
#barchk('SibSp')
#print(type([train]))
#print(type(train))
#print(type(train))
#print(type([train]))
#print([train][0])
#print([train]['Name'])
#print([train,test])
#print(type(train))
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
#print(train['Title'].value_counts())
title_Mapping = {'Mr':0, 'Miss':1,'Mrs':2,'Master':3,'Dr':3,'Rev':3,'Mlle':3,'Col':3,'Major':3,'Jonkheer':3,'Sir':3,'Ms':3,'Countess':3,'Lady':3,'Don':3,'Mme':3,'Capt':3}
train['Title']=train['Title'].map(title_Mapping)
train.drop('Name',axis=1,inplace=True)
sex_Mapping = {'male':0, 'female':1}
train['Sex']=train['Sex'].map(sex_Mapping)
train['Age'].fillna(train.groupby('Title')['Age'].transform("median"),inplace=True)
train.loc[train['Age']<=16 ,'Age']=0
train.loc[(16<train['Age']) & (train['Age']<=26),'Age']=1
train.loc[(26<train['Age']) & (train['Age']<=36),'Age']=2
train.loc[(36<train['Age']) & (train['Age']<=62),'Age']=3
train.loc[62<train['Age'],'Age']=4
#barchk('Age')

facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()

#plt.show()

a = train[train['Pclass']==1]['Embarked'].value_counts()
b = train[train['Pclass']==2]['Embarked'].value_counts()
c = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([a,b,c])
df.index=['1st C','2st C','3st C']
#print(df)
#df.plot(kind='bar',stacked=True)
#plt.show()
#train.fillna('S',inplace=True)
train['Embarked'].fillna('S',inplace=True)
train['Embarked']=train['Embarked'].map({'S':0,'C':1,'Q':2})
#print(train)
train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'),inplace=True)
#print(train)

#fillna('Embarked')
#train[train['Age']<16]
#print(train[train['Age']<16])

facet = sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Fare',shade=True)
facet.set(xlim=(0,train['Fare'].max()))
#facet.set(xlim=(0,20))
facet.add_legend()
#plt.show()


train.loc[train['Fare']<=17,'Fare']=0
train.loc[(17<train['Fare']) & (train['Fare']<=30),'Fare']=1
train.loc[(30<train['Fare']) & (train['Fare']<=100),'Fare']=2
train.loc[100<train['Fare'],'Fare']=3
#print(train.head())

#print(train['Cabin'].value_counts())
train['Cabin']=train['Cabin'].str[:1]
#print(train)
#print(a)
a1 = train[train['Pclass']==1]['Cabin'].value_counts()
#print(a)
b2 = train[train['Pclass']==2]['Cabin'].value_counts()
c3 = train[train['Pclass']==3]['Cabin'].value_counts()
df2 = pd.DataFrame([a1,b2,c3])
df2.index=['1st C','2st C','3st C']
df2.plot(kind='bar',stacked=True)
#plt.show()

train['Cabin']=train['Cabin'].map({'A':0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'F':2.0,'G':2.4,'T':2.8})
train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'),inplace=True)
#print(train)

train['Familysize']=train['SibSp']+train['Parch']+1
#print(train)

facet = sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Familysize',shade=True)
facet.set(xlim=(0,train['Familysize'].max()))
#facet.set(xlim=(20,30))
facet.add_legend()
#plt.xlim(0)
#plt.show()
train['Familysize']=train['Familysize'].map({1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2.0,7:2.4,8:2.8,9:3.2,10:3.6,11:4.0,})
train.drop(['Ticket','SibSp','Parch','PassengerId'],axis=1,inplace=True)
#print(train.head(10))

train_data = train.drop(['Survived'],axis=1)
#print(train_data)
target = train['Survived']
train.info()

#print(train['Title'])
#barchk('Title')
#print(train)
# #dataset['Title']=dataset['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
# for dataset in [train]:
#     print(type(dataset))
#     #print(dataset['Sex'])
#     #print(type(dataset['Name']))
#     dataset['Title']=dataset['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)

#print(dataset)
#print(barchk("Title"))

#barchk("Pclass")
#barchk("Sex")
#barchk("SibSp")
#barchk("Parch")
#barchk("Embarked")

'''
barchk("Age")
barchk("Ticket")
barchk("Fare")
'''
#print(survived)
#print(test.head())
#print(test.info())
#print(train.isnull().sum())




test = pd.read_csv('test.csv')
#test1 = pd.read_csv('C:\\Users\\kimji\\Desktop\\20190109_Kaggle\\test1.csv')
#test = pd.read_csv('C:\\Users\\kimji\\Desktop\\20190109_Kaggle\\test.csv')
#head=test.head()
#print(head)
#print(test.head())
#print(test[test['Survived']==1])
#print(type(test[list('')]))
#print(test[list('true')])
#print(test1[test1['1']])
#print(list([['true','he']['ha','hu']]))


#Survived = test[test['Survived'] == 1]['Sex'].value_counts()
#print(Survived)
#print(type(Survived))

def barchk(name):
    survived = test[test['Survived']==1][name].value_counts()
    dead = test[test['Survived']==0][name].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index=['survived','dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))
    plt.show()
    #print(df.filter(like='Mr.',axis=1))
    #print(df)
#barchk('SibSp')
#print(type([test]))
#print(type(test))
#print(type(test))
#print(type([test]))
#print([test][0])
#print([test]['Name'])
#print([test,test])
#print(type(test))
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
#print(test['Title'].value_counts())
title_Mapping = {'Mr':0, 'Miss':1,'Mrs':2,'Master':3,'Dr':3,'Rev':3,'Mlle':3,'Col':3,'Major':3,'Jonkheer':3,'Sir':3,'Ms':3,'Countess':3,'Lady':3,'Don':3,'Mme':3,'Capt':3, 'Dona':3}
test['Title']=test['Title'].map(title_Mapping)
test.drop('Name',axis=1,inplace=True)
sex_Mapping = {'male':0, 'female':1}
test['Sex']=test['Sex'].map(sex_Mapping)
test['Age'].fillna(test.groupby('Title')['Age'].transform("median"),inplace=True)
test.loc[test['Age']<=16 ,'Age']=0
test.loc[(16<test['Age']) & (test['Age']<=26),'Age']=1
test.loc[(26<test['Age']) & (test['Age']<=36),'Age']=2
test.loc[(36<test['Age']) & (test['Age']<=62),'Age']=3
test.loc[62<test['Age'],'Age']=4
#barchk('Age')



#plt.show()

a = test[test['Pclass']==1]['Embarked'].value_counts()
b = test[test['Pclass']==2]['Embarked'].value_counts()
c = test[test['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([a,b,c])
df.index=['1st C','2st C','3st C']
#print(df)
#df.plot(kind='bar',stacked=True)
#plt.show()
#test.fillna('S',inplace=True)
test['Embarked'].fillna('S',inplace=True)
test['Embarked']=test['Embarked'].map({'S':0,'C':1,'Q':2})
#print(test)
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'),inplace=True)
#print(test)

#fillna('Embarked')
#test[test['Age']<16]
#print(test[test['Age']<16])




test.loc[test['Fare']<=17,'Fare']=0
test.loc[(17<test['Fare']) & (test['Fare']<=30),'Fare']=1
test.loc[(30<test['Fare']) & (test['Fare']<=100),'Fare']=2
test.loc[100<test['Fare'],'Fare']=3
#print(test.head())

#print(test['Cabin'].value_counts())
test['Cabin']=test['Cabin'].str[:1]
#print(test)
#print(a)
a1 = test[test['Pclass']==1]['Cabin'].value_counts()
#print(a)
b2 = test[test['Pclass']==2]['Cabin'].value_counts()
c3 = test[test['Pclass']==3]['Cabin'].value_counts()
df2 = pd.DataFrame([a1,b2,c3])
df2.index=['1st C','2st C','3st C']
df2.plot(kind='bar',stacked=True)
#plt.show()

test['Cabin']=test['Cabin'].map({'A':0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'F':2.0,'G':2.4,'T':2.8})
test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'),inplace=True)
#print(test)

test['Familysize']=test['SibSp']+test['Parch']+1
#print(test)


#plt.xlim(0)
#plt.show()
test['Familysize']=test['Familysize'].map({1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2.0,7:2.4,8:2.8,9:3.2,10:3.6,11:4.0,})
test.drop(['Ticket','SibSp','Parch'],axis=1,inplace=True)
#print(test.head(10))


























from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

clf = SVC(gamma='auto')
scoring = 'accuracy'
score = cross_val_score(clf,train_data,target,cv=k_fold,n_jobs=1,scoring=scoring)
print(score)
print(round(np.mean(score)*100,2))

clf.fit(train_data,target)

test_data = test.drop('PassengerId',axis=1).copy()
prediction = clf.predict(test_data)
#print(prediction)
submission = pd.DataFrame({"PassengerId":test['PassengerId'],'Survived':prediction})
submission.to_csv('submission.csv',index=False)