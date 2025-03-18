import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.inspection import PartialDependenceDisplay

dragon_model = joblib.load('dragon_model.pkl')
scaler = joblib.load('dragon_scaler.pkl')

data = {
    'FIP': [3.300939],
    'ERA': [3.38],
    'P_BABIP': [0.286468],
    'P_BB%': [0.077940],
    'OPS': [0.668],
    'OBP': [0.315],
    'ISO': [0.097],
    'H_K%': [0.191627],
    'Defense%': [0.982],
    'wOBA': [0.295344],
    'WHIP': [1.512000]
}
predict_data = pd.DataFrame(data)

# 特徵
features = ['FIP', 'ERA', 'P_BABIP', 'P_BB%', 'OPS', 'OBP', 'ISO', 'H_K%', 'Defense%', 'wOBA', 'WHIP']

# 範圍
features_range = {
	'FIP': np.linspace(3.5, 4, 5),
    'ERA': np.linspace(3.5, 4, 5),
    'P_BABIP': np.linspace(0.28, 0.36, 5),
    'P_BB%': np.linspace(0.5, 1.5, 5),
    'OPS':  np.linspace(0.74, 0.78, 5),
    'OBP':  np.linspace(0.2, 0.4, 5),
    'ISO':  np.linspace(0.06, 0.1, 5),
    'H_K%': np.linspace(0.05, 0.2, 5),
    'Defense%': np.linspace(0.8, 0.9, 5),
    'wOBA': np.linspace(0.327, 0.340, 5),
    'WHIP': np.linspace(1.0, 1.4, 5)
}

# 預測結果
predictions_dict = {}

# 靈敏度分析，對每個特徵的不同值進行預測
for feature in features:
    feature_values = features_range[feature]  # 獲取當前特徵的範圍
    predictions = []

    for value in feature_values:    
        temp_data = predict_data.copy()# 複製預測資料並修改指定特徵值
        temp_data[feature] = value  # 將當前特徵值設為新值

        # 標準化處理
        temp_data_scaled = scaler.transform(temp_data)
        temp_data_scaled = pd.DataFrame(temp_data_scaled, columns=temp_data.columns)

        # 預測並存儲
        prediction = dragon_model.predict(temp_data_scaled)

        # 預測並存儲結果
        prediction = dragon_model.predict(temp_data_scaled)
        predictions.append(prediction[0])

    predictions_dict[feature] = predictions

    # 繪製每個特徵的邊際效應圖
    plt.figure(figsize=(8, 6))
    plt.plot(feature_values, predictions, marker='o', linestyle='-', color='b')
    plt.xlabel(feature)
    plt.ylabel('Predicted Win Rate')
    plt.title(f'Marginal Effect of {feature} on Predicted Win Rate')
    plt.grid(True)
    plt.show()

'''
predict_data = {
    'FIP': [3.300939],
    'ERA': [3.38],
    'P_BABIP': [0.286468],
    'P_BB%': [0.077940],
    'OPS': [0.668],
    'OBP': [0.315],
    'ISO': [0.097],
    'H_K%': [0.191627],
    'Defense%': [0.982],
    'wOBA': [0.295344],
    'WHIP': [1.512000]
}
predict_data_df = pd.DataFrame(predict_data)

predict_data_scaled = scaler.transform(predict_data_df)

predictions = dragon_model.predict(predict_data_scaled)

print("假如味全龍球員個人成績都與去年相當，那麼2025年勝率預計是")
print(predictions)'''
