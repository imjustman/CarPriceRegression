from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import joblib
import time
import tqdm
import os


dir_data_path = Path('./Data')
latest_data_file = max([f for f in dir_data_path.glob('*.csv') if 'pre' not in f.name], key=lambda p: p.stat().st_mtime)

dir_model_path = Path('./Model')
latest_model_file = max(dir_model_path.glob('*.pkl'), key=lambda p: p.stat().st_mtime)

now = datetime.now(ZoneInfo("Asia/Seoul"))
date_str = now.strftime('%Y_%m_%d')

df = pd.read_csv(latest_data_file)

def make_rare_label(df):
    df_copy = df.copy()
    rare_label_index = {}
    for col in ['Badge', 'Model']:
        count = df_copy[col].value_counts()
        count = 100 * count / len(df_copy)
        rare_label = count[count < 0.1].index
        rare_label_index[col] = rare_label

        df_copy.loc[df_copy[col].isin(rare_label), col] = 'Others'
    return rare_label_index, df_copy

def remove_outliers(df, col):
    df_copy = df.copy()
    Q1 = df_copy[col].quantile(0.25)
    Q3 = df_copy[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_clean = df_copy[(df_copy[col] <= upper) & (df_copy[col] >= lower)]

    return df_clean

rare_label_index, df = make_rare_label(df)

joblib.dump(rare_label_index, f"./Artifacts/category_map_{date_str}.pkl")

df = remove_outliers(df, 'Price')

df['Year'] = df['Year'].astype('int')
df['Year'] = df['Year'].astype('str')
df['Model_Year'] = df['Year'].apply(lambda x: x[:4]).astype(int)
df['Model_Month'] = df['Year'].apply(lambda x: x[4:]).astype(int)
df['Vehicle_Age'] = time.localtime().tm_year - df['Model_Year']
df.drop(['Year', 'Model_Year'], axis=1, inplace=True)

df.to_csv(f"./Data/pre_encar_data_{date_str}.csv", index=False, encoding='utf-8-sig')

X = df.drop(['Price', 'Id'], axis=1)
y = df['Price']

train_X, valid_X, train_y, valid_y = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
model = joblib.load(latest_model_file)
model.set_params(preprocess__target_encoder__targetencoder__smooth=25.0)
model.set_params(preprocess__target_encoder__targetencoder__categories='auto')

model.fit(train_X, train_y)

preds = model.predict(valid_X)

rmse = np.sqrt(mean_squared_error(valid_y, preds))
mae = mean_absolute_error(valid_y, preds)
r2 = r2_score(valid_y, preds)

file_path = os.path.join('./Results', f"metrics_{date_str}.txt")
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(f"RMSE : {rmse:.4f}\n")
    f.write(f"MAE : {mae:.4f}\n")
    f.write(f"R2 : {r2:.4f}\n")

model.fit(X, y)

joblib.dump(model, f"./Model/model_pipe_{date_str}.pkl")
