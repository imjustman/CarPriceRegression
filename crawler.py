from tqdm import tqdm
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import pandas as pd
import time
import os

year = time.localtime().tm_year
month = time.localtime().tm_mon
day = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%d')

file_name = 'encar_data' + f'_{year}_{month}_{day}.csv'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36'
}

temp_data=[]

url = f'https://api.encar.com/search/car/list/premium?count=true&q=(And.Hidden.N._.CarType.Y.)&sr=%7CModifiedDate%7C0%7C20'
response = requests.get(url, headers=headers)
cnt = response.json()['Count']

for i in tqdm(range(0, cnt, 50)):
    try:
        url = f'https://api.encar.com/search/car/list/premium?count=true&q=(And.Hidden.N._.CarType.Y.)&sr=%7CModifiedDate%7C{i}%7C50'
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            print(f"\n[오류] {i}지점에서 응답 없음 (Status: {response.status_code})")
            time.sleep(10)
            continue

        res = response.json().get('SearchResults', [])

        for item in res:
            row = {
                'Id': item.get('Id'),
                'HomeService': None,
                'EncarDiagnosis': None,
                'Manufacturer': item.get('Manufacturer'),
                'Model': item.get('Model'),
                'Badge': item.get('Badge'),
                'Transmission': item.get('Transmission'),
                'FuelType': item.get('FuelType'),
                'Year': item.get('Year'),
                'Mileage': item.get('Mileage'),
                'Price': item.get('Price'),
                'SellType': item.get('SellType'),
                'OfficeCityState': item.get('OfficeCityState')
            }

            marks = item.get('ServiceMark', [])

            if len(marks) > 1:
                row['EncarDiagnosis'] = marks[1]
                row['HomeService'] = marks[0]
            elif len(marks) == 1:
                if marks[0] == 'EncarMeetgo':
                    row['HomeService'] = marks[0]
                else:
                    row['EncarDiagnosis'] = marks[0]

            temp_data.append(row)

            if (i > 0) and (i % 10000 == 0):
                df_chunk = pd.DataFrame(temp_data)

                if not os.path.exists(f'./Data/{file_name}.csv'):
                    df_chunk.to_csv(f'./Data/{file_name}.csv', index=False, encoding='utf-8-sig')
                else:
                    df_chunk.to_csv(f'./Data/{file_name}.csv', index=False, encoding='utf-8-sig', mode='a', header=False)

                temp_data=[]
                time.sleep(0.5)
    except Exception as e:
        print(f"\n[예외 발생] {i} 지점에서 중단 {e}")

        if temp_data:
            pd.DataFrame(temp_data).to_csv(f'./Data/{file_name}.csv', index=False, encoding='utf-8-sig', mode='a', header=not os.path.exists(f'./Data/{file_name}.csv'))
print('데이터 수집 및 저장 완료.')

