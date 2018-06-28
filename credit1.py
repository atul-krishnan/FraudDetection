import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import classification_report,accuracy_score 
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


data=pd.read_csv('credit_card.csv')

#print "Types of fraudulent transactions are:"
#print list(data.loc[data.isFraud==1].type.drop_duplicates().values)

data = data.loc[(data.type == 'CASH_OUT')|(data.type == 'TRANSFER')]#----only these two types of transactions are fraudulent. Limited data to it
data.loc[data.type == 'TRANSFER', 'type'] = 0 #encoding char value as an int so that ML algorithm can use it
data.loc[data.type == 'CASH_OUT', 'type'] = 1 # same as above line
data = data.drop(['nameOrig','nameDest','isFlaggedFraud'],axis = 1)#dropped the features that don't affect the outcome. 
data=data.sample(frac=0.6,random_state=1)

print data.shape


#Determining the no. of fraud cases in dataset

Fraud = data[data['isFraud']==1]
Valid = data[data['isFraud']==0]

outlier_fraction = len(Fraud)/float(len(Valid))
print (outlier_fraction)

print ('Fraud cases : {}'.format(len(Fraud)))
print ('Valid cases : {}'.format(len(Valid)))

#corelation matrix

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
#print x.head(5)

x = np.array(x)
y = np.array(y)

#print shape of x and y

print(x.shape)
print(y.shape)

# !--- Training the model---!

#defining the random state
state=1

#defining the outlier detecton methods

classifiers={
	"Isolation Forest":IsolationForest(max_samples=100,contamination=outlier_fraction,random_state=state),

	"Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,contamination=outlier_fraction)
}

#Fit the model

n_outliers=len(Fraud)

for i,(clf_name,clf) in enumerate(classifiers.items()):
	

    
	if clf_name=="Local Outlier Factor":
		y_pred=clf.fit_predict(x)
		scores_pred=clf.negative_outlier_factor_ 

	else:
	 	clf.fit(x)
		scores_pred=clf.decision_function(x)
		y_pred=clf.predict(x)

	#reshape theprediction values to 0 for valid and 1 for fraud

	y_pred[y_pred==1]=0
	y_pred[y_pred==-1]=1

	n_errors=(y_pred!=y).sum()

	#run classification metrics

	print ('{}'.format(clf_name,n_errors))
	print(accuracy_score(y,y_pred))
	print(classification_report(y,y_pred))
