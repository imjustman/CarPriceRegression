from pathlib import Path

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# dir_data_path = str(max(Path('./Data').glob('*.csv'), key=lambda p: p.stat().st_mtime))
dir_data_path = str(max([f for f in Path('./Data').glob('*.csv') if 'pre' not in f.name], key=lambda p: p.stat().st_mtime))

df = pd.read_csv(dir_data_path)
print(df.columns)
st.set_page_config(page_title='ML Model Predictor', layout='wide')

st.title("중고차 가격 예측 서비스 (Encar 기반)")
st.markdown("""
이 대시보드는 수집된 데이터를 바탕으로 학습된 모델을 사용하여 중고차의 적정 가격을 예측합니다.
왼쪽 사이드바에서 차량의 정보를 입력하세요.
""")

with st.sidebar:
    st.header('차량 정보 입력')

    HomeService = st.selectbox('엔카 믿고 등록 여부', ['엔카믿고', '없음'])
    EncarDiagnosis = st.selectbox('엔카 진단 등록 여부', ['엔카진단', '엔카진단+', '엔카진단++', '없음'])
    brand = st.selectbox("브랜드", df['Manufacturer'].unique().tolist())

    model = st.selectbox('모델', df.loc[df['Manufacturer'] == brand, 'Model'].unique().tolist() + ['기타'])

    badge_list = []
    if model == '기타':
        badge_list = ['기타']
    else:
        badge_list = df.loc[df['Model'] == model, 'Badge'].unique().tolist()

    badge = st.selectbox('배지', badge_list + ['기타'])

    transmission = st.selectbox('변속기', df['Transmission'].unique().tolist())
    fuel_type = st.selectbox('연료', df['FuelType'].unique().tolist())
    sell_type = st.selectbox('판매방식', df['SellType'].unique().tolist())
    office_city = st.selectbox('지역', df['OfficeCityState'].unique().tolist())

    model_year = str(st.slider('연식', 2000, 2026, 2013))
    model_month = st.slider('월', 1, 12, 6)
    model_month = f"{model_month:02d}"
    model_year_month = model_year + model_month
    Year = float(model_year_month + ".0")


    mileage = float(st.number_input('주행거리 (km)', min_value=0, value=300000, step=1000))


    predict_button = st.button('가격 예측하기')

col1, col2 = st.columns([1, 1])

if predict_button:
    input_data = {
        'HomeService': HomeService,
        'EncarDiagnosis': EncarDiagnosis,
        'Manufacturer': brand,
        'Model': model,
        'Badge': badge,
        'Transmission': transmission,
        'FuelType': fuel_type,
        'Year': Year,
        'Mileage': mileage,
        'SellType': sell_type,
        'OfficeCityState': office_city
    }

    try:
        with st.spinner('AI 모델이 가격을 계산 중입니다'):
            # response = requests.post('http://localhost:8003/predict', json=input_data)
            response = requests.post('http://localhost:8000/predict', json=input_data)
            response.raise_for_status()
            result = response.json()

            prediction = int(float(result.get('predict')[1:-1]))
            prediction_id = result.get('prediction_Id')

        with col1:
            st.success("### 예측 결과")
            st.metric(label='예상 판매 가격', value=f"{prediction}만원")

        with col2:
            st.info("### 예측 근거")
            importance_df = pd.DataFrame({
                '특성': result.get('explain_feature_labels'),
                '기여도': result.get('explain_feature_values')
            })

            fig = px.bar(importance_df.sort_values(by='기여도', ascending=False), x="기여도", y="특성", orientation='h', title="예측 근거")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"서버 연결 실패: {e}")

else:
    st.info('왼쪽 사이드바에서 정보를 입력하고 "가격 예측하기" 버튼을 눌러주세요.')




