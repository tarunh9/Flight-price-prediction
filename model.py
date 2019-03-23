import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('Data_Train.csv')
test = pd.read_csv('Test_set.csv')

print("\n Number of cells with missing values : ", train.isnull().sum())
print("\n Number of cells with missing values : ", test.isnull().sum())

train = train.dropna()

# Cleaning Date_of_Journey
train['Journey_Day'] = pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.day
train['Journey_Month'] = pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.month
train = train.drop('Date_of_Journey', axis = 1)

test['Journey_Day'] = pd.to_datetime(test.Date_of_Journey, format='%d/%m/%Y').dt.day
test['Journey_Month'] = pd.to_datetime(test.Date_of_Journey, format='%d/%m/%Y').dt.month
test = test.drop('Date_of_Journey', axis = 1)

# Cleaning Duration
## Training set
duration = list(train['Duration'])

for i in range(len(duration)) :
    if len(duration[i].split()) != 2:
        if 'h' in duration[i] :
            duration[i] = duration[i].strip() + ' 0m'
        elif 'm' in duration[i] :
            duration[i] = '0h {}'.format(duration[i].strip())

dur_hours = []
dur_minutes = []  

for i in range(len(duration)) :
    dur_hours.append(int(duration[i].split()[0][:-1]))
    dur_minutes.append(int(duration[i].split()[1][:-1]))
    
train['Duration_hours'] = dur_hours
train['Duration_minutes'] = dur_minutes

train = train.drop('Duration', axis = 1)

## Test set
durationT = list(test['Duration'])

for i in range(len(durationT)) :
    if len(durationT[i].split()) != 2:
        if 'h' in durationT[i] :
            durationT[i] = durationT[i].strip() + ' 0m'
        elif 'm' in durationT[i] :
            durationT[i] = '0h {}'.format(durationT[i].strip())
            
dur_hours = []
dur_minutes = []  

for i in range(len(durationT)) :
    dur_hours.append(int(durationT[i].split()[0][:-1]))
    dur_minutes.append(int(durationT[i].split()[1][:-1]))
  
    
test['Duration_hours'] = dur_hours
test['Duration_minutes'] = dur_minutes

test = test.drop('Duration', axis = 1)

train['Depart_time_hour'] = pd.to_datetime(train.Dep_Time).dt.hour
train['Depart_time_minutes'] = pd.to_datetime(train.Dep_Time).dt.minute

train['Arr_time_hour'] = pd.to_datetime(train.Arrival_Time).dt.hour
train['Arr_time_minutes'] = pd.to_datetime(train.Arrival_Time).dt.minute

train = train.drop(['Arrival_Time','Dep_Time'], axis = 1)

test['Depart_time_hour'] = pd.to_datetime(test.Dep_Time).dt.hour
test['Depart_time_minutes'] = pd.to_datetime(test.Dep_Time).dt.minute

test['Arr_time_hour'] = pd.to_datetime(test.Arrival_Time).dt.hour
test['Arr_time_minutes'] = pd.to_datetime(test.Arrival_Time).dt.minute

test = test.drop(['Arrival_Time','Dep_Time'], axis = 1)

X_train = train.iloc[:, train.columns != 'Price'].values
y_train = train.iloc[:,6].values
X_test = test.iloc[:,:].values

# LabelEncoder
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
le2 = LabelEncoder()

X_train[:,0] = le1.fit_transform(X_train[:,0])
X_train[:,1] = le1.fit_transform(X_train[:,1])
X_train[:,2] = le1.fit_transform(X_train[:,2])
X_train[:,3] = le1.fit_transform(X_train[:,3])
X_train[:,4] = le1.fit_transform(X_train[:,4])
X_train[:,5] = le1.fit_transform(X_train[:,5])


X_test[:,0] = le2.fit_transform(X_test[:,0])
X_test[:,1] = le2.fit_transform(X_test[:,1])
X_test[:,2] = le2.fit_transform(X_test[:,2])
X_test[:,3] = le2.fit_transform(X_test[:,3])
X_test[:,4] = le2.fit_transform(X_test[:,4])
X_test[:,5] = le2.fit_transform(X_test[:,5])

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
y_train = y_train.reshape((len(y_train), 1)) 
y_train = sc.fit_transform(y_train)


# SVM
from sklearn.svm import SVR
svr = SVR(kernel = "rbf")
svr.fit(X_train,y_train)
y_pred = sc.inverse_transform(svr.predict(X_test))
pd.DataFrame(y_pred, columns = ['Price']).to_excel("Predictions.xlsx", index = False)


# Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = svr,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_



