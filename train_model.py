import pandas as pd
import numpy as np
import seaborn as sns
import xgboost
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import joblib
import matplotlib.pyplot as plt

# 1️⃣ Đọc dữ liệu
df = pd.read_csv("D:/DemoWebXGBoost/data/bodyPerformance.csv")

# 2️⃣ Mã hóa cột class thành 0, 1, 2, 3
le = LabelEncoder()
df["class_encoded"] = le.fit_transform(df["class"])

# 3️⃣ Mã hóa cột gender
df["gender_encoded"] = df["gender"].map({"M": 1, "F": 0})

# 4️⃣ Xử lý dữ liệu đầu vào
X = df.drop(columns=["class", "class_encoded", "gender"])
y = df["class_encoded"]

# 5️⃣ Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6️⃣ Cân bằng dữ liệu với SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 7️⃣ Chia train-test
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 8️⃣ Tìm kiếm siêu tham số tối ưu với GridSearchCV
param_grid = {
    'n_estimators': [100],
    'max_depth': [7],
    'learning_rate': [0.2],
    'min_child_weight': [3]
}

xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=4,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    xgb, param_grid, cv=5, scoring='f1_macro', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV f1_macro:", grid_search.best_score_)

# 9️⃣ Lấy mô hình tốt nhất
best_model = grid_search.best_estimator_

# Đánh giá mô hình tốt nhất
y_pred = best_model.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
print("\nClassification Report:")
print(df_report)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC-AUC macro
y_test_bin = pd.get_dummies(y_test)
y_score = best_model.predict_proba(X_test)
roc_auc_macro = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")
print("ROC-AUC macro:", roc_auc_macro)

# Lưu mô hình tốt nhất và scaler
joblib.dump(best_model, "D:/DemoWebXGBoost/models/xgboost_model.pkl")
joblib.dump(scaler, "D:/DemoWebXGBoost/models/scaler.pkl")
