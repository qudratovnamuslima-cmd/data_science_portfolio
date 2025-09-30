import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    'Лошадинные силы': [150, 120, 180, 210, 160],
    'Расход топлива': [8, 6, 10, 11, 9]
})

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print(data_scaled)


pca = PCA(n_components=1)
principal_components = pca.fit_transform(data_scaled)
print(principal_components)

"""
Задание: У вас есть данные о клиентах банка, включая их возраст, доход,
кредитный рейтинг, количество открытых счетов и сумму задолженности.
Ваша задача — использовать PCA для снижения размерности данных и
построить модель классификации, которая предсказывает вероятность дефолта.
Данные:
Возраст.
Доход.
Кредитный рейтинг.
Количество открытых счетов.
Сумма задолженности.
Метка (0 — нет дефолта, 1 — дефолт).
Задачи:
Загрузите данные и выполните стандартизацию признаков.
Примените PCA, чтобы сократить размерность данных и сохранить 95% информации.
Постройте модель логистической регрессии на данных с уменьшенной размерностью и оцените её точность.
Сравните результаты с моделью, обученной на исходных данных без PCA.
"""


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from time import time

data = pd.read_excel('credit_data (2).xlsx')
x = data.drop('Метка', axis=1)
y = data['Метка']


scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)
print(x_scaler)

pca = PCA()
pca.fit(x_scaler)

pca = PCA(n_components=3)
x_pca = pca.fit_transform(x_scaler)

x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.3, random_state=42)

start_pca = time()
model = LogisticRegression()
model.fit(x_train, y_train)
finished_pca = time()
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели (PCA: {accuracy})')
print(f'Модель обучилась за {finished_pca-start_pca:.3f} секунд')
print(f'Точность модели с PCA: {round(accuracy*100, 2)}%')

x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, test_size=0.3, random_state=42)
start_pca = time()
model = LogisticRegression()
model.fit(x_train, y_train)
finished_pca = time()
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели на стандартизированных данных: {round(accuracy*100, 2)}%')  # 93.33%
print(f'Модель обучилось за {finished_pca-start_pca:.3f} секунд')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
start_pca = time()
model = LogisticRegression()
model.fit(x_train, y_train)
finished_pca = time()
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели на исходных данных: {round(accuracy*100, 2)}%')
print(f'Модель обучилось за {finished_pca-start_pca:.3f} секунд')

