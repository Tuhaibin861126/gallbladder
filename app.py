# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 设置页面配置 (必须是第一个 Streamlit 命令)
st.set_page_config(page_title="Gallbladder Polyp Prediction", layout="wide")

# 加载资源 (现在在 set_page_config 之后)
@st.cache_resource
def load_assets():
    model = joblib.load('gallbladder_model.pkl')
    scaler = joblib.load('gallbladder_scaler.pkl')
    numerical_features = joblib.load('gallbladder_numerical_features.pkl')
    categorical_features = joblib.load('gallbladder_categorical_features.pkl')
    return model, scaler, numerical_features, categorical_features

model, scaler, numerical_features, categorical_features = load_assets()


st.title('Gallbladder Polyp Prediction Model')

# 输入表单
input_data = {}
col1, col2, col3 = st.columns(3)

with col1:
    input_data['diameter'] = st.number_input('Diameter (mm):', min_value=0.0, format="%.2f")
    input_data['NLR'] = st.number_input('NLR:', min_value=0.0, format="%.2f")

with col2:
    # 分类变量处理（确保与训练时编码一致）
    stalk_mapping = {"Yes": 1, "No": 0}
    input_data['stalk'] = st.selectbox('Stalk Present:', options=list(stalk_mapping.keys()))

with col3:
    input_data['age'] = st.number_input('Age (years):', min_value=0, format="%d")
    input_data['CA199'] = st.number_input('CA199 (U/mL):', min_value=0.0, format="%.2f")

# 预测逻辑
if st.button('Predict'):
    try:
        # 转换分类变量
        input_data['stalk'] = stalk_mapping[input_data['stalk']]

        # 创建包含所有特征的DataFrame（顺序：数值特征在前，分类在后）
        df_numerical = pd.DataFrame([input_data])[numerical_features]
        df_categorical = pd.DataFrame([input_data])[categorical_features]

        # 标准化数值特征
        scaled_numerical = scaler.transform(df_numerical)

        # 合并数据
        final_input = np.concatenate([scaled_numerical, df_categorical.values], axis=1)

        # 预测
        probability = model.predict_proba(final_input)[0, 1]

        # 显示结果
        st.success(f'Predicted Probability of Neoplastic Polyp: **{probability:.1%}**')

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Debug Info - Numerical Features:", numerical_features) # 调试信息
        st.write("Debug Info - Categorical Features:", categorical_features) #调试信息
