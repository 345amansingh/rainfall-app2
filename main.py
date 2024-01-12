from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import pandas as pd
data = pd.read_csv("seattle-weather.csv")
data.head()

countrain = len(data[data.weather == 'rain'])

countsun = len(data[data.weather == 'sun'])

countdrizzle = len(data[data.weather == 'drizzle'])

countsnow = len(data[data.weather == 'snow'])

countfog = len(data[data.weather == 'fog'])


print('percent of rain:{:2f}%'.format((countrain/(len(data.weather))*100)))

print('percent of sun:{:2f}%'.format((countsun/(len(data.weather))*100)))

print('percent of drizzle:{:2f}%'.format((countdrizzle/(len(data.weather))*100)))

print('percent of snow:{:2f}%'.format((countsnow/(len(data.weather))*100)))

print('percent of fog:{:2f}%'.format((countfog/(len(data.weather))*100)))


data[['precipitation', 'temp_max', 'temp_min', 'wind']].describe()


data.isna().sum()


data=data.drop(['date'], axis=1)
Q1 = data[['precipitation', 'temp_max', 'temp_min', 'wind']].quantile(0.25)
Q3 = data[['precipitation', 'temp_max', 'temp_min', 'wind']].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data[['precipitation', 'temp_max', 'temp_min', 'wind']] < (Q1 - 1.5 * IQR)) | (data[['precipitation', 'temp_max', 'temp_min', 'wind']] > (Q3 + 1.5 * IQR))).any(axis=1)]


import numpy as np
data.precipitation = np.sqrt(data.precipitation)
data.wind=np.sqrt(data.wind)

data.head()

lc = LabelEncoder()
data['weather'] = lc.fit_transform(data['weather'])

data.head()


x = ((data.loc[:,data.columns!='weather']).astype(int)).values[:,0:]
y = data['weather'].values

data.weather.unique()


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=2)

knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
print('KNN accuracy:{:.2f}%'.format(knn.score(x_test,y_test)*100))

xgb = XGBClassifier()
xgb.fit(x_train, y_train)

print('XGB accuracy:{:.2f}%'.format(xgb.score(x_test, y_test) * 100))

gbc=GradientBoostingClassifier(subsample=0.5,n_estimators=450,max_depth=5,max_leaf_nodes=25)
gbc.fit(x_train,y_train)
print('GBC accuracy:{:.2f}%'.format(gbc.score(x_test,y_test)*100))


input=[[1.140175,8.9,2.8,2.469818]]

ot=xgb.predict(input)

print('the weather is:')
if(ot==0):
  print('Drizzle')
elif (ot==1):
  print('fogg')
elif (ot==2):
  print('rain')
elif (ot==3):
  print('snow')
else:
  print('sun')


import pickle
file = 'model.pkl'
pickle.dump(xgb, open(file, 'wb'))