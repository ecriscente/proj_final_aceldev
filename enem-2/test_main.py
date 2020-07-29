import pandas as pd
import numpy as np
from sklearn import metrics

enem_train = pd.read_csv("train.csv")

enem_test = pd.read_csv("test.csv")

corr = enem_train.corr().reset_index()

features_corr = []
features = []

for index, item in zip(corr["index"], corr["NU_NOTA_MT"]):
    if (item <  -0.15 or item > 0.15):
        features.append(index)


features.pop(1)
features.pop(0)
features.pop(-1)
features.pop(-1)
features.pop(4)

features_corr = list(features)

features.pop(3)

enem_train= enem_train.loc[(enem_train['NU_NOTA_CN'].notnull()) & (enem_train['NU_NOTA_CN']!=0) &
                  (enem_train['NU_NOTA_CH'].notnull()) & (enem_train['NU_NOTA_CH']!=0) &
                  (enem_train['NU_NOTA_LC'].notnull())&(enem_train['NU_NOTA_LC']!=0) &
                  (enem_train['NU_NOTA_REDACAO'].notnull()) & (enem_train['NU_NOTA_REDACAO']!=0)]

enem_test= enem_test.loc[(enem_test['NU_NOTA_CN'].notnull()) & (enem_test['NU_NOTA_CN']!=0) &
                  (enem_test['NU_NOTA_CH'].notnull()) & (enem_test['NU_NOTA_CH']!=0) &
                  (enem_test['NU_NOTA_LC'].notnull())&(enem_test['NU_NOTA_LC']!=0) &
                  (enem_test['NU_NOTA_REDACAO'].notnull()) & (enem_test['NU_NOTA_REDACAO']!=0)]

enem_test['NU_NOTA_COMP1'].fillna(0,inplace=True)
enem_test['NU_NOTA_COMP2'].fillna(0,inplace=True)
enem_test['NU_NOTA_COMP3'].fillna(0,inplace=True)
enem_test['NU_NOTA_COMP4'].fillna(0,inplace=True)
enem_test['NU_NOTA_COMP5'].fillna(0,inplace=True)

enem_train['NU_NOTA_COMP1'].fillna(0,inplace=True)
enem_train['NU_NOTA_COMP2'].fillna(0,inplace=True)
enem_train['NU_NOTA_COMP3'].fillna(0,inplace=True)
enem_train['NU_NOTA_COMP4'].fillna(0,inplace=True)
enem_train['NU_NOTA_COMP5'].fillna(0,inplace=True)

y_train = enem_train['NU_NOTA_MT']

x_train = enem_train[features]
x_test = enem_test[features]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train, y_train)

predict_train = regressor.predict(x_train)

predict_train = pd.Series(predict_train)

predict_note = regressor.predict(x_test)

predict_note = pd.Series(predict_note)

num_inscricao = enem_test['NU_INSCRICAO']

answer = pd.concat([num_inscricao, predict_note], axis=1)

answer.rename(columns={0:'NU_NOTA_MT'}, inplace=True)

answer.to_csv('answer.csv',columns=['NU_INSCRICAO', 'NU_NOTA_MT'], sep=',', index=False)

def test_0():
    answer = pd.read_csv("answer.csv")
    assert answer.shape == (4223, 2) 
    assert set(["NU_INSCRICAO", "NU_NOTA_MT"]) == set(answer.columns)
