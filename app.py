import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load mô hình và scaler
model = joblib.load('D:/DemoWebXGBoost/models/xgboost_model.pkl')
scaler = joblib.load('D:/DemoWebXGBoost/models/scaler.pkl')

# Tiêu đề
st.title("Dự đoán Tình trạng Sức khỏe")

st.write("""
Giúp phân loại tình trạng sức khỏe (A, B, C, D) dựa trên các chỉ số y tế:
- Nhập các chỉ số như tuổi, BMI, cholesterol, thể lực, nhịp tim...
- Nhấn nút Dự đoán để xem kết quả.
""")

# Các chỉ số đầu vào mẫu (có thể chỉnh theo dữ liệu thực tế của bạn)
age = st.number_input("Tuổi (Từ 18 đến 70)", min_value=18, max_value=70, value=30)
height = st.number_input("Chiều cao (cm) (Từ 140 đến 210 )", min_value=140, max_value=210, value=170)
weight = st.number_input("Cân nặng (kg) (Từ 40 đến 150 )", min_value=40, max_value=150, value=70)
body_fat = st.number_input("Tỷ lệ mỡ cơ thể (%) (Từ 5.0 đến 60.0 )", min_value=5.0, max_value=60.0, value=20.0)
diastolic = st.number_input("Huyết áp tâm trương (mmHg) (Từ 50 đến 150 )", min_value=50, max_value=150, value=80)
systolic = st.number_input("Huyết áp tâm thu (mmHg) (Từ 80 đến 200 )", min_value=80, max_value=200, value=120)
grip_force = st.number_input("Lực tay (kg) (Từ 0 đến 70 )", min_value=0, max_value=70, value=35)
sit_bend_forward = st.number_input("Độ gập người (cm) (Từ -20 đến 50 )", min_value=-20, max_value=50, value=10)
sit_ups_count = st.number_input("Số lần gập bụng (Từ 0 đến 100 )", min_value=0, max_value=100, value=30)
broad_jump = st.number_input("Bật xa (cm) (Từ 100 đến 300 )", min_value=100, max_value=300, value=200)
gender = st.selectbox("Giới tính", options=["Nam", "Nữ"])

# Mã hóa biến giới tính
gender_encoded = 1 if gender == "Nam" else 0

# Tạo DataFrame từ input
input_data = pd.DataFrame({
    'age': [age],
    'height_cm': [height],
    'weight_kg': [weight],
    'body fat_%': [body_fat],
    'diastolic': [diastolic],
    'systolic': [systolic],
    'gripForce': [grip_force],
    'sit and bend forward_cm': [sit_bend_forward],
    'sit-ups counts': [sit_ups_count],
    'broad jump_cm': [broad_jump],
    'gender_encoded': [gender_encoded]
})

# Dự đoán khi nhấn nút
if st.button("Dự đoán"):
    # Chuẩn hóa input
    input_data_scaled = scaler.transform(input_data)

    # Dự đoán
    y_pred = model.predict(input_data_scaled)
    y_prob = model.predict_proba(input_data_scaled)

    # Map nhãn sang tên lớp (A, B, C, D)
    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    predicted_label = label_map[int(y_pred[0])]

    st.success(f"Tình trạng sức khỏe dự đoán: **{predicted_label}**")
    st.write("Xác suất dự đoán từng nhóm:")
    prob_df = pd.DataFrame(y_prob, columns=['A', 'B', 'C', 'D'])
    st.dataframe(prob_df.style.format("{:.2%}"))
