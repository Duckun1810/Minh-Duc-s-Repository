## Các bước để chạy code

Bước 1: Mở Terminal, thiết lập môi trưởng ảo (nếu cần)
```
.\venv\Scripts\activate
```

Bước 2: Cài đặt các gói thư viện cần thiết

```bash
pip install -r requirements.txt
```
Bước 3: Huấn luyện mô hình XGBoost (chờ khoảng 5 phút hoặc hơn)
```
cd notebooks
```
```python
python train_model.py
```

Bước 4: Mở giao diện

```python
streamlit run app.py
```
## Lưu ý
1. Chỉ cần huấn luyện mô hình một lần. Kết quả mô hình sẽ được lưu lại cho các lần chạy  sau.
2. Thay đổi đường dẫn trong code của file train_model.py và app.py phù hợp với thiết bị của bạn.
