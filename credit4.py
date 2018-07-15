import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import classification_report,accuracy_score #determine how successful we are in our outlier detection
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix

data=pd.read_csv('credit_card.csv')

#data=data.sample(frac=0.1,random_state=1)

#Exploratory Data Analysis

print(data.type.value_counts())

f, ax = plt.subplots(1, 1, figsize=(12, 9))
data.type.value_counts().plot(kind='bar', title="Transaction type", ax=ax, figsize=(12,9))
plt.show()

ax = data.groupby(['type', 'isFraud']).size().plot(kind='bar')
ax.set_title("# of transaction which are the actual fraud per transaction type")
ax.set_xlabel("(Type, isFraud)")
ax.set_ylabel("Count of transaction")
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))

f, bx = plt.subplots(1, 1, figsize=(12, 9))

bx = data.groupby(['type', 'isFlaggedFraud']).size().plot(kind='bar')
bx.set_title("# of transaction which is flagged as fraud per transaction type")
bx.set_xlabel("(Type, isFlaggedFraud)")
bx.set_ylabel("Count of transaction")
for p in bx.patches:
    bx.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))



#From the above histogram we could remove the unnecessary features that won't affect our analysis
print "From the above graphs we can get the fraudulent transactions :"
print list(data.loc[data.isFraud==1].type.drop_duplicates().values)

data = data.loc[(data.type == 'CASH_OUT')|(data.type == 'TRANSFER')]#----only these two types of transactions are fraudulent. Limited data to it
data.loc[data.type == 'TRANSFER', 'type'] = 0 #encoding char value as an int so that ML algorithm can use it
data.loc[data.type == 'CASH_OUT', 'type'] = 1 # same as above line
data = data.drop(['nameOrig','nameDest',],axis = 1)
# when oldBalanceDest = newBalDest = 0, it is a strong indicator of fraud. Again, assigning a specific numeric value to those two columns in all database entries where the above condition is satisfied

data.loc[(data.oldbalanceDest == 0) & (data.newbalanceDest == 0) & (data.amount != 0),['oldbalanceDest', 'newbalanceDest']] = -1

# when oldBalOrig = newBalOrig = 0, it is a sign of non-fraud. Assigning a numeric value to those two columns.
data.loc[(data.oldbalanceOrg == 0) & (data.newbalanceOrig == 0) & (data.amount != 0),['oldbalanceOrg', 'newbalanceOrig']] = -2

#Feature Engineering
data['errorBalanceOrig'] = data.newbalanceOrig + data.amount - data.oldbalanceOrg

F = data[data['isFraud']==1]
V = data[data['isFraud']==0]

print ('Fraud cases : {}'.format(len(F)))
print ('Valid cases : {}'.format(len(V)))


#corelation heatmap

cormat=data.corr()
fig=plt.figure(figsize=(12,9))

sns.heatmap(cormat,vmax=.8,square=True)
plt.show()

#Get all the columns from the data frame

columns=data.columns.tolist()

#filtering 

columns=[c for c in columns if c not in ['isFraud']]

#store the variabe we'll be predicting on

target='isFraud'

x=data[columns]
y=data[target]
print "\n"


#print shape of x and y

print("The shape of x and y:")
print(x.shape)
print(y.shape)

#Startfiied Shuffle Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
for train_index, test_index in sss.split(x,y):
	#print("TRAIN:", train_index, "TEST:", test_index)
	trainX, testX = x.iloc[train_index], x.iloc[test_index]
	trainY, testY = y.iloc[train_index], y.iloc[test_index]
	

Fraud = testY[testY==1]

Valid = testY[testY==0]



outlier_fraction = len(Fraud)/float(len(Valid))
print "Outlier fraction is:"
print outlier_fraction


# !--- Training the model---!

#defining the random state
state=1

#AdaBoost

clf = AdaBoostClassifier(n_estimators=50,random_state = state,learning_rate = 1.0)
clf.fit(trainX,trainY)
pred = clf.predict(testX)
print "Confusion Matrix of AdaBoost is :-\n"
print confusion_matrix(testY,pred)
print "\n\nClassification report for AdaBoost:-"
print classification_report(testY,pred)

#Isolation Forest


clf=IsolationForest(contamination=outlier_fraction,random_state=state,n_jobs=4)
testX = testX.drop(['errorBalanceOrig',],axis = 1)
clf.fit(testX)
scores_pred=clf.decision_function(testX)
y_pred=clf.predict(testX)

#reshape theprediction values to 0 for valid and 1 for fraud

y_pred[y_pred==1]=0
y_pred[y_pred==-1]=1

n_errors=(y_pred!=testY).sum()
#run classification metrics

#print ('{}'.format(clf_name,n_errors))
#print(accuracy_score(y,y_pred))  #since it's an unbalanced class problem the accurancy score will be inappropriate 
print "Classification report for Isolation Forest:-"
print(classification_report(testY,y_pred))



