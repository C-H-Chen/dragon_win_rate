import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import f_regression
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# 讀取 Excel 檔案
file_path = "CPBL數據 _面試用.xlsx"
xls = pd.ExcelFile(file_path)
	
# 讀取工作表
team_performance_df = pd.read_excel(xls, sheet_name='球隊成績')
team_pitching_df = pd.read_excel(xls, sheet_name='球隊投手')
team_hitting_df = pd.read_excel(xls, sheet_name='球隊打擊')
team_defense_df = pd.read_excel(xls, sheet_name='球隊守備')
player_hitting_df = pd.read_excel(xls, sheet_name='球員打擊(2010-24)')
player_pitching_df = pd.read_excel(xls, sheet_name='球員投手(2010-24)')
player_defense_df = pd.read_excel(xls, sheet_name='球員守備(2010-24)')

# 特徵工程:
# 處理年度格式，判斷是否為上下年度
team_performance_df['年度'] = team_performance_df['年度'].apply(lambda x: str(x).split('(')[0].strip() if '(' in str(x) else str(x))

# 篩選年份範圍
team_performance_df = team_performance_df[team_performance_df['年度'].astype(int).between(2010, 2024)]
team_pitching_df = team_pitching_df[team_pitching_df['年度'].astype(int).between(2010, 2024)]
team_defense_df = team_defense_df[team_defense_df['年度'].astype(int).between(2010, 2024)]

# 轉換年度為字串
team_performance_df['年度'] = team_performance_df['年度'].astype(str)
team_pitching_df['年度'] = team_pitching_df['年度'].astype(str)
team_hitting_df['年度'] = team_hitting_df['年度'].astype(str)
team_defense_df['年度'] = team_defense_df['年度'].astype(str)
player_pitching_df['年度'] = player_pitching_df['年度'].astype(str)
player_hitting_df['年度'] = player_hitting_df['年度'].astype(str)
player_defense_df['年度'] = player_defense_df['年度'].astype(str)

# 計算全年
team_performance_df_agg = team_performance_df.groupby(['球隊', '年度']).agg(
    全年勝率=('勝率', lambda x: x.sum() / 2),
	全年主場勝=('主場勝', 'sum'),
    全年主場敗=('主場敗', 'sum'),
	全年客場勝=('客場勝', 'sum'),
	全年客場敗=('客場敗', 'sum')
).reset_index()
#print(team_performance_df_agg[['球隊', '年度', '全年勝率']])

avg_era = team_pitching_df.groupby('年度')['防禦率'].mean()
avg_fip = team_pitching_df.groupby('年度').apply(lambda x: ((13 * x['被全壘打'] + 3 * x['四壞球'] - 2 * x['奪三振']) / x['局數'] + 3.10).mean())
avg_ops = team_hitting_df.groupby('年度').apply(lambda x: (x['上壘率'] + x['長打率']).mean())

# 球隊成績
team_performance_df_agg['Home-Away_diff'] = (team_performance_df_agg['全年主場勝'] - team_performance_df_agg['全年主場敗']) - (team_performance_df_agg['全年客場勝'] - team_performance_df_agg['全年客場敗']) 

#球隊投手
team_pitching_df['FIP'] = ((13 * team_pitching_df['被全壘打'] + 3 * team_pitching_df['四壞球'] - 2 * team_pitching_df['奪三振']) / team_pitching_df['局數'] + 3.10)# / team_pitching_df['年度'].map(avg_fip))
team_pitching_df['P_BABIP'] = (team_pitching_df['被安打'] - team_pitching_df['被全壘打']) / (team_pitching_df['面對打席'] - team_pitching_df['被全壘打'] - team_pitching_df['奪三振'] - team_pitching_df['四壞球'])
team_pitching_df['ERA+'] = 100 * (team_pitching_df['年度'].map(avg_era) / team_pitching_df['防禦率'])
team_pitching_df['K%'] = team_pitching_df['奪三振'] / team_pitching_df['面對打席']
team_pitching_df['P_BB%'] = team_pitching_df['四壞球'] / team_pitching_df['面對打席']
team_pitching_df['K/BB'] = (team_pitching_df['奪三振'] / team_pitching_df['四壞球'])
team_pitching_df['HR%'] = team_pitching_df['被全壘打'] / team_pitching_df['面對打席']

#球隊打者
team_hitting_df['OPS'] = (team_hitting_df['上壘率'] + team_hitting_df['長打率'])# / team_hitting_df['年度'].map(avg_ops))
team_hitting_df['H_K%'] = team_hitting_df['三振'] / team_hitting_df['打數']
team_hitting_df['SH/AB'] = team_hitting_df['犧牲短打'] / team_hitting_df['打數']
team_hitting_df['SB/G'] = team_hitting_df['盜壘成功'] / team_hitting_df['出賽數']
team_hitting_df['ISO'] = team_hitting_df['長打率'] - team_hitting_df['打擊率']
team_hitting_df['PPG'] = team_hitting_df['得分'] - team_hitting_df['出賽數']

# 球員打擊
player_hitting_df_agg = player_hitting_df.groupby(['球隊', '年度']).agg(
    #TB=('壘打數', 'sum'),
    #DP=('雙殺打', 'sum'),
	PA=('打席', 'sum'),
	AB=('打數', 'sum'),
    SH=('犧短', 'sum'),
	SF=('犧飛', 'sum'),
    one=('一安', 'sum'),
    double=('二安', 'sum'),
    triple=('三安', 'sum'),
    HR=('全壘打', 'sum'),
    BB=('四壞球', 'sum'),
    HBP=('死球', 'sum'),
	K=('被三振', 'sum'),
    #H_GB_FB=('滾飛出局比', 'mean'),
    SB_percent=('盜壘率', 'mean'),
    #OPS=('整體攻擊指數', 'mean'),
    #Silver_percent=('銀棒指數', 'mean')
).reset_index()
player_hitting_df_agg['SH%'] = (player_hitting_df_agg['SH'] + player_hitting_df_agg['SF']) / player_hitting_df_agg['PA']
player_hitting_df_agg['wOBA'] = ((0.69 * player_hitting_df_agg['BB'] + 0.72 * player_hitting_df_agg['HBP'] 
                               + 0.9 * player_hitting_df_agg['one'] + 1.25 * player_hitting_df_agg['double']
							   + 1.6 * player_hitting_df_agg['triple'] + 1.8 * player_hitting_df_agg['HR']) 
							   / (player_hitting_df_agg['AB'] + player_hitting_df_agg['BB'] + player_hitting_df_agg['SF'] + player_hitting_df_agg['HBP']))
player_hitting_df_agg['H_BB%'] = player_hitting_df_agg['BB'] / player_hitting_df_agg['AB']
player_hitting_df_agg['H_K/BB'] = player_hitting_df_agg['K'] / player_hitting_df_agg['BB']

# 球員投手		   
player_pitching_df_agg = player_pitching_df.groupby(['球隊', '年度']).agg(
    WHIP=('每局被上壘率', 'mean'),
	HBP=('死球', 'sum'),
	#GB=('滾地出局', 'sum'),
    #ERA=('防禦率', 'mean'),
    #P_HRs=('被全壘打', 'sum'),
    #P_K=('奪三振', 'sum'),
    #P_GB_FB=('滾飛出局比', 'mean')
).reset_index()

#球員守備
player_defense_df_agg = player_defense_df.groupby(['球隊', '年度']).agg(
        #PO=('刺殺', 'sum'),
        #A=('助殺', 'sum'),
        #DP=('雙殺', 'sum'),
		SB =('被盜成功', 'sum'),
        CS=('盜壘阻殺', 'sum'),
		#Fielding_percent =('守備率', 'mean')
).reset_index()
player_defense_df_agg['CS%'] = player_defense_df_agg['CS'] /(player_defense_df_agg['SB'] + player_defense_df_agg['CS'])
player_defense_df_agg = player_defense_df_agg.merge(team_pitching_df[['球隊', '年度', '局數']], on=['球隊', '年度'], how='left')
player_defense_df_agg['SBA'] = (player_defense_df_agg['CS'] + (player_defense_df_agg['SB']) / player_defense_df_agg['局數'])

#球隊守備
team_defense_df['DP/DO'] = team_defense_df['雙殺'] / team_defense_df['守備機會']
team_defense_df = team_defense_df.merge(player_pitching_df_agg[['球隊', '年度', 'HBP']], on=['球隊', '年度'], how='left')
team_defense_df = team_defense_df.merge(team_pitching_df[['球隊', '年度', '四壞球', '面對打席', '被全壘打', '被安打']], on=['球隊', '年度'], how='left')
team_defense_df['DER'] = 1 - ((team_defense_df['被安打'] + team_defense_df['失誤'] - team_defense_df['被全壘打']) 
                           / (team_defense_df['面對打席'] - team_defense_df['四壞球'] - team_defense_df['HBP'] - team_defense_df['被全壘打']))                    


# 合併資料
df = team_performance_df_agg[['球隊', '年度', '全年勝率']].rename(columns={'全年勝率': 'Win_Rate'}).copy()#, 'Home-Away_diff'
df = df.merge(team_pitching_df[['球隊', '年度', 'FIP', '防禦率', 'P_BABIP', 'P_BB%']].rename(columns={'防禦率': 'ERA'})#, 'K/BB', 'ERA+', 'HR%', 'K%'
         , on=['球隊', '年度'], how='left')
df = df.merge(team_hitting_df[['球隊', '年度', 'OPS', '上壘率', 'ISO','H_K%']]#
         .rename(columns={'上壘率': 'OBP'}), on=['球隊', '年度'], how='left')
df = df.merge(team_defense_df[['球隊', '年度', '守備率']], on=['球隊', '年度'],how='left').rename(columns={'守備率': 'Defense%'})#, 'DER'
df = df.merge(player_hitting_df_agg[['球隊', '年度', 'wOBA']], on=['球隊', '年度'], how='left')#, 'H_K/BB', 'SH%'
df = df.merge(player_pitching_df_agg[['球隊', '年度', 'WHIP']], on=['球隊', '年度'], how='left')#
#df = df.merge(player_defense_df_agg[['球隊', '年度', 'SBA']], on=['球隊', '年度'], how='left')
df = df.drop_duplicates()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(df)
#print(df.isnull().sum())

# 確保資料是按照年份排序
df = df.sort_values(by=['年度'])

# 確定訓練集和測試集的分割點
train_size = int(len(df) * 0.7)
x_train, x_test = df.iloc[:train_size].drop(columns=['球隊', '年度', 'Win_Rate']), df.iloc[train_size:].drop(columns=['球隊', '年度', 'Win_Rate'])
y_train, y_test = df.iloc[:train_size]['Win_Rate'], df.iloc[train_size:]['Win_Rate']

# 分割資料為訓練集和測試集
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 標準化
scaler = StandardScaler()  # 定義 StandardScaler
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)

#特徵相關性
correlation_matrix = x_train_scaled.select_dtypes(include=[float, int]).corr()
print(correlation_matrix)
plt.tight_layout()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

y_col = y_train 
y_col_name = y_col.name if isinstance(y_col, pd.Series) else y_col


all_columns = x_train_scaled.select_dtypes(include=[float, int]).columns.tolist()  # 確保選擇的是數值型欄位

# 計算每個 x 與固定的 y 之間的偏相關
results = []
for feature in all_columns:

    pcorr_result = pg.partial_corr(data=x_train_scaled.join(y_col), x=feature, y=y_col_name)
    results.append(pcorr_result)
    print(f"偏相關分析: 特徵 {feature} 與 {y_col_name} 的結果: \n{pcorr_result}\n")

results_df = pd.concat(results, ignore_index=True)
print(results_df)


#F score
f_scores, p_values = f_regression(x_train, y_train)

# 顯示F分數和P值
for name, f_score, p_value in zip(all_columns, f_scores, p_values):
    print(f"{name}: F-Score = {f_score:.4f}, P-value = {p_value:.4f}")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# F分數條形圖
ax[0].bar(all_columns, f_scores, color='skyblue')
ax[0].set_title('F-Score for Each Feature')
ax[0].set_xlabel('Features')
ax[0].set_ylabel('F-Score')
# P值條形圖
ax[1].bar(all_columns, p_values, color='lightcoral')
ax[1].set_title('P-Value for Each Feature')
ax[1].set_xlabel('Features')
ax[1].set_ylabel('P-Value')

plt.tight_layout()


# 斜率
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

# 輸出特徵係數
slopes = lin_reg.coef_
all_columns = x_train.columns  
for name, slope in zip(all_columns, slopes):
    print(f"{name}: {slope:.4f}")
# 視覺化
plt.bar(all_columns, slopes, color='skyblue')
plt.xlabel('Features')
plt.ylabel('Coefficient (Slope)')
plt.title('Linear Regression Coefficients (Slopes)')


# 假設檢定
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)

# Elastic Net Logistic Regression
elastic_net = ElasticNetCV(cv=5, l1_ratio=0.5, max_iter=10000)
elastic_net.fit(x_train_scaled, y_train)

# 使用statsmodels進行假設檢定
x_train_const = sm.add_constant(x_train_scaled)  # 添加截距
# 確保索引一致
y_train = y_train.reset_index(drop=True)
x_train_const = x_train_const.reset_index(drop=True)

logit_model = sm.Logit(y_train, x_train_const)
result = logit_model.fit()

# OLS 檢定結果摘要
print(result.summary())

# 提取回歸係數、P 值、置信區間
coef = result.params[1:]  # 排除截距項
p_values = result.pvalues[1:]
conf = result.conf_int().iloc[1:]  # 排除截距項的置信區間
conf.columns = ['Lower CI', 'Upper CI']

# 創建DataFrame存儲特徵、係數、P 值、置信區間
df = pd.DataFrame({
    'Feature': x_train.columns,
    'Coefficient': coef,
    'P-Value': p_values,
    'Lower CI': conf['Lower CI'],
    'Upper CI': conf['Upper CI']
})

# 回歸係數與置信區間圖表
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# 條形圖顯示回歸係數
sns.barplot(x='Coefficient', y='Feature', data=df, palette='coolwarm', errorbar=None, hue='Coefficient', legend=False)

# 橫向置信區間
for i in range(df.shape[0]):
    plt.plot([df['Lower CI'].iloc[i], df['Upper CI'].iloc[i]], [i, i], color='black')
# 標記顯著性 (P 值 < 0.05)
for i in range(len(df)):
    if df['P-Value'].iloc[i] < 0.05:
        plt.text(df['Coefficient'].iloc[i], i, '*', color='red', va='center')

# 標題與軸標籤
plt.title('Regression Hypothesis Tests for Elastic Net Logistic Regression Model')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.axvline(0, color='black', linewidth=0.8)  # 中心線

plt.tight_layout()

# 檢測多重共線性
# 計算每個特徵的 VIF
vif_data = pd.DataFrame()
vif_data["feature"] = x_train.columns  # 設定特徵名稱
vif_data["VIF"] = [variance_inflation_factor(x_train_scaled, i) for i in range(x_train_scaled.shape[1])]
print(vif_data)

plt.show()
