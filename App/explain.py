import streamlit as st
import shap

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

def compute_shap_values(pipeline, input_df):
    X_transformed = pipeline.named_steps['preprocess'].transform(input_df)

    model = pipeline.named_steps['model']

    explainer = get_explainer(model)
    shap_values = explainer(X_transformed)[0].values.tolist()
    feature_labels = pipeline.named_steps['preprocess'].feature_names_in_.tolist()

    return feature_labels, shap_values