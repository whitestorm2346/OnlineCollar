import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 假設你已經提取了很多特徵向量，每個向量代表一張圖像
# 這些向量和對應的標籤用於訓練分類模型
features = np.random.rand(100, 128)  # 模擬的特徵向量數據 (100個樣本, 每個樣本128維)
labels = np.random.randint(0, 10, 100)  # 模擬的標籤數據 (10個類別)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = SVC()

param_grid = {
    'C': [0.1, 1, 10, 100], 
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 
    'gamma': ['scale', 'auto'],  
    'degree': [2, 3, 4] 
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("最佳參數：", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("分類報告：")
print(classification_report(y_test, y_pred))

joblib.dump(best_model, '../models/face_recognition_model.pkl')
