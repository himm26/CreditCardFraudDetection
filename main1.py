
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.preprocessing import StandardScaler, MaxAbsScaler
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#data Path
DATA_PATH = "/home/himanshu/Downloads/DATA/creditcardfraud/"



data = pd.read_csv(DATA_PATH + 'creditcard.csv')

# print(data.head())

feature = ['V%d' % number for number in range(1,29)] + ['Amount']

target = 'Class'

X = data[feature]
y = data[target]

# X = MaxAbsScaler(X)
scaler = StandardScaler()

scaler.fit(X)

# X.hist(figsize=(20,20))
# plt.show()

#splitting data

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=101)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test,y_pred))





