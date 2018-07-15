# FraudDetection
Fraud detection has now become a very important aspect in corporate as well as in personal life . With improvement in machine learning day by day it has become easier in detecting fraud without human intervention . But as someone said "With great power comes great responsibility" , likewise for a good machine learning model we need a good dataset . The dataset here has been downloaded from https://www.kaggle.com/ntnu-testimon/paysim1/data . (Note: After downloading the dataset just rename the csv file to credit_card.csv)
The detection technique that has been used in here is Anomaly Detection .
I have commented in most places for the better understanding of the code .

Before running the code make sure you have following packages installed in your system
 1. Pandas
 2. Numpy
 3. Matplotlib(Used for the representation purpose)
 4. Seaborn(For heatmap generation) 
 5. Sklearn(For the implementation of the models)
 
 In credit1.py , the detection methods used are LocalOutlierFactor and Isolation Forest . But since these both methods give a very low f1 score it is not recommended to implement . 
 In credit4.py file , the anomaly detection methods used are AdaBoost and Isolation Forest . After running the program we could observe that the AdaBoost gives a much higher value of f1 score than that of the Isolation Forest . 
 

