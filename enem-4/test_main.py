import pandas as pd

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

dftrain = pd.read_csv('train.csv')
dftest = pd.read_csv('test.csv')


x = dftrain[['TP_ST_CONCLUSAO', 'NU_IDADE', 'TP_ANO_CONCLUIU', 'TP_ESCOLA']]
y = dftrain['IN_TREINEIRO']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.01)

random_forest = RandomForestRegressor(n_estimators = 100)
random_forest.fit(x_train, y_train)

Xres = dftest[['TP_ST_CONCLUSAO', 'NU_IDADE', 'TP_ANO_CONCLUIU', 'TP_ESCOLA']]

random_forest = RandomForestRegressor(n_estimators = 100)
random_forest.fit(x, y)

Yres = random_forest.predict(Xres)


dftest['IN_TREINEIRO'] = np.around(Yres, 0)

dftest = dftest[['NU_INSCRICAO','IN_TREINEIRO']]

dftest.to_csv('answer.csv', index=False)


def test_0():
    answer = pd.read_csv("answer.csv")
    assert answer.shape == (4570, 2) 
    assert set(["NU_INSCRICAO", "IN_TREINEIRO"]) == set(answer.columns)
