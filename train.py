import pandas as pd
from pathlib import Path
import joblib
import time
import tqdm

dir_data_path = Path('./Data')
latest_data_file = max(dir_data_path.glob('*.csv'), key=lambda p: p.stat().st_mtime)

dir_model_path = Path('./Model')
latest_model_file = max(dir_model_path.glob('*.pkl'), key=lambda p: p.stat().st_mtime)

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

joblib.dump(rare_label_index, f'./Artifacts/category_map_{time.localtime().tm_year}_{time.localtime().tm_mon}_{time.localtime().tm_mday}.pkl')

df = remove_outliers(df, 'Price')

df['Year'] = df['Year'].astype('int')
df['Year'] = df['Year'].astype('str')
df['Model_Year'] = df['Year'].apply(lambda x: x[:4]).astype(int)
df['Model_Month'] = df['Year'].apply(lambda x: x[4:]).astype(int)
df['Vehicle_Age'] = time.localtime().tm_year - df['Model_Year']
df.drop(['Year', 'Model_Year'], axis=1, inplace=True)

X = df.drop(['Price', 'Id'], axis=1)
y = df['Price']

model = joblib.load(latest_model_file)
model.set_params(preprocess__target_encoder__targetencoder__smooth=25.0)
model.set_params(preprocess__target_encoder__targetencoder__categories='auto')

model.fit(X, y)

joblib.dump(model, f'./Model/model_pipe_{time.localtime().tm_year}_{time.localtime().tm_mon}_{time.localtime().tm_mday}.pkl')
