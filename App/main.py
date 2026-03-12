import streamlit as st
import requests
import pandas as pd
import plotly.express as px

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
    brand = st.selectbox("브랜드", ['기아', '현대', '제네시스', 'KG모빌리티(쌍용)', '쉐보레(GM대우)', '르노코리아(삼성)', '기타 제조사'])
    model = st.selectbox('모델', ['카니발 4세대'])
    badge = st.selectbox('배지', ['가솔린 9인승 프레스티지', '9인승 노블레스', '9인승 시그니처'])
    transmission = st.selectbox('변속기', ['오토', '세미오토', '수동', 'CVT'])
    fuel_type = st.selectbox('연료', ['가솔린', '디젤', '가솔린+전기', 'LPG(일반인 구입)', '전기', '수소', '가솔린+LPG', 'LPG+전기'])
    sell_type = st.selectbox('판매방식', ['일반', '렌트', '리스'])
    office_city = st.selectbox('지역', ['경기', '서울', '부산', '대구', '광주', '대전', '인천', '전북', '충남', '경남', '충북', '울산', '경북', '전남', '강원', '제주'])

    model_year = str(st.slider('연식', 2000, 2026, 2013))
    model_month = st.slider('월', 1, 12, 6)
    model_month = f"{model_month:02d}"
    model_year_month = model_year + model_month
    Year = float(model_year_month + ".0")


    mileage = float(st.number_input('주행거리 (km)', min_value=0, value=200000, step=1000))


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
        with st.spinner('AI 모델이 가격을 계산 중입니다....'):
            response = requests.post('http://localhost:8000/predict', json=input_data)
            response.raise_for_status()
            result = response.json()

            prediction = int(float(result.get('predict')[1:-1]))
            prediction_id = result.get('prediction_Id')
            print(result)
        with col1:
            st.success("### 예측 결과")
            st.metric(label='예상 판매 가격', value=f"{prediction}만원")

        with col2:
            st.info("### 특성 중요도 (Feature Importance)")
            importance_df = pd.DataFrame({
                'Feature': ['연식', '주행거리', '브랜드', '뱃지여부'],
                'Importance': [0.45, 0.3, 0.15, 0.1]
            })
            fig = px.bar(importance_df, x="Importance", y="Feature", orientation='h', title="예측 근거")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"서버 연결 실패: {e}")

else:
    st.info('왼쪽 사이드바에서 정보를 입력하고 "가격 예측하기" 버튼을 눌러주세요.')




