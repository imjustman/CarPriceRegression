import joblib

category_map = joblib.load('./Artifacts/category_map.pkl')
model = joblib.load('./Model/model_pipe.pkl')

print(model)