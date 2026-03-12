import pandas as pd
import time

def preprocess(df, category_map):

    if df['Model'].item() not in category_map['Model']:
        df['Model'] = 'Others'

    if df['Badge'].item() not in category_map['Badge']:
        df['Badge'] = 'Others'

    df['Year'] = df['Year'].astype(int)
    df['Year'] = df['Year'].astype(str)

    df['Model_Year'] = df['Year'].apply(lambda x: x[:4]).astype(int)
    df['Model_Month'] = df['Year'].apply(lambda x: x[4:]).astype(int)
    df['Vehicle_Age'] = time.localtime().tm_year - df['Model_Year']

    df.drop(['Year', 'Model_Year'], axis=1, inplace=True)

    if df['HomeService'].item() == '없음':
        df['HomeService'] = 'None'
    else:
        df['HomeService'] = 'EncarMeetgo'

    if df['EncarDiagnosis'].item() == '엔카진단':
        df['EncarDiagnosis'] = 'EncarDiagnosisP0'
    elif df['EncarDiagnosis'].item() == '엔카진단+':
        df['EncarDiagnosis'] = 'EncarDiagnosisP1'
    elif df['EncarDiagnosis'].item() == '엔카진단++':
        df['EncarDiagnosis'] = 'EncarDiagnosisP2'
    else:
        df['EncarDiagnosis'] = 'None'


    return df