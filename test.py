from time import time
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

test_path = 'test'
os.makedirs(test_path, exist_ok=True)

new_data = pd.read_csv('IBM_Stock_1990_2025.csv').iloc[:2530].copy()
print(new_data.info())
print('columns:', new_data.columns.tolist())

new_data['Date'] = pd.to_datetime(new_data['Date'], errors='coerce')
new_data = new_data.sort_values('Date').reset_index(drop=True)

for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
    if col in new_data.columns:
        new_data[col] = new_data[col].astype(str).str.replace(',', '').astype(float)

print(new_data.dtypes)
print(new_data.head())

print(new_data.columns.tolist())

new_data = new_data.drop(columns=[c for c in ['Dividends', 'Stock Splits'] if c in new_data.columns])
print(new_data.isnull().sum())
print(new_data.head(10))
print(new_data.describe())

new_data['ma_5'] = new_data['Adj Close'].rolling(window=5).mean()
new_data['ma_10'] = new_data['Adj Close'].rolling(window=10).mean()
new_data['ma_20'] = new_data['Adj Close'].rolling(window=20).mean()
new_data['return_1'] = new_data['Adj Close'].pct_change(1)
new_data['return_5'] = new_data['Adj Close'].pct_change(5)
new_data['lag_1'] = new_data['Adj Close'].shift(1)
new_data['lag_5'] = new_data['Adj Close'].shift(5)

new_data = new_data.dropna().reset_index(drop=True)


feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                'ma_5', 'ma_10', 'ma_20',
                'return_1', 'return_5',
                'lag_1', 'lag_5']
feature_cols = [c for c in feature_cols if c in new_data.columns]

X = new_data[feature_cols]
y = new_data['Adj Close']

start = time()
model = joblib.load('random_forest_model.pkl')
prediction = model.predict(X)
finish = time()
print(f'Конец предсказания модели RandomForestRegressor на новых данных за {finish - start:.2f} сек.')

MSE = mean_squared_error(y, prediction)
RMSE = MSE ** 0.5
MAE = mean_absolute_error(y, prediction)
R2 = r2_score(y, prediction)

print("\nОценка модели на новых данных:")
print(f"MSE  = {MSE:.4f}")
print(f"RMSE = {RMSE:.4f}")
print(f"MAE  = {MAE:.4f}")
print(f"R²   = {R2:.4f}")


result = new_data[['Date', 'Adj Close']].copy()
result['Prediction'] = prediction
result.to_csv(os.path.join(test_path, 'ibm_predictions.csv'), index=False)

print(f"\nРезультаты сохранены в {os.path.join(test_path, 'ibm_predictions.csv')}")









