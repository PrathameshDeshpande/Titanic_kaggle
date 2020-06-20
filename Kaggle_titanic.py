import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

dataset = pd.read_csv("train.csv")
test =pd.read_csv("test.csv")
dataset.loc[dataset['Cabin'].notnull(), 'Cabin'] = 1
dataset.loc[dataset['Cabin'].isnull(), 'Cabin'] = 0
dataset.loc[dataset['Embarked'].isnull(), 'Embarked'] = 'C'
test.loc[test['Cabin'].notnull(), 'Cabin'] = 1
test.loc[test['Cabin'].isnull(), 'Cabin'] = 0
test.loc[test['Embarked'].isnull(), 'Embarked'] = 'C'
y_train=dataset.iloc[:,1:2].values
cols = list(dataset.columns)
X_train=dataset[cols[4:8]+cols[10:11]+cols[11:12]]
###############################################
col = list(test.columns)
test=test[col[3:7]+col[9:10]+col[10:11]]
###############################################
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer(
            [('encoder', OneHotEncoder(), [0])],
            remainder='passthrough')
X_train = columnTransformer.fit_transform(X_train)
X_train=X_train[:,1:]
columnTransformer = ColumnTransformer(
            [('encoder', OneHotEncoder(), [5])],
            remainder='passthrough')
X_train = columnTransformer.fit_transform(X_train)
X_train=X_train[:,1:]
###################################################
columnTransformer = ColumnTransformer(
           [('encoder', OneHotEncoder(), [0])],
           remainder='passthrough')
test = columnTransformer.fit_transform(test)
test=test[:,1:]
columnTransformer = ColumnTransformer(
           [('encoder', OneHotEncoder(), [5])],
           remainder='passthrough')
test = columnTransformer.fit_transform(test)
test=test[:,1:]
###################################
from sklearn.impute import SimpleImputer
imputer =SimpleImputer(missing_values= np.nan,strategy='mean')
imputer=imputer.fit(X_train[:,3:4])
X_train[:,3:4]=imputer.transform(X_train[:,3:4])
##########################################
imputer =SimpleImputer(missing_values= np.nan,strategy='mean')
imputer=imputer.fit(test[:,3:4])
test[:,3:4]=imputer.transform(test[:,3:4])
#######################################################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
test = sc.transform(test)


N, D = X_train.shape
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(D,)),
  tf.keras.layers.Dense(10,activation ='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
from keras import optimizers

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

r = model.fit(X_train, y_train,epochs=1000)

y_pred=model.predict(test)
y_pred = np.round(y_pred)


















