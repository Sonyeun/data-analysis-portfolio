####################### 결측치 처리 #######################

# 데이터셋 파일 불러오기
df = pd.read_csv('C:\\Users\\User\\Desktop\\원드라이\\정동은\\2024\\다변량분석\\Multiple Linear Regression\\과제시작1\\data_happy_v1.CSV')

# AQ26A, AQ26A_1, AQ26A_2, AQ26A_3, AQ5_1, AQ5_2 열의 결측치를 0으로 채우기
cols_to_fill = ['AQ26A', 'AQ26A_1', 'AQ26A_2', 'AQ26A_3', 'AQ5_1', 'AQ5_2']
df[cols_to_fill] = df[cols_to_fill].fillna(0)

# 그 외의 열에서 결측치가 있는 행 제거
df.dropna(inplace=True)

# 데이터셋 파일로 저장 (선택사항)
df.to_csv('C:\\Users\\User\\Desktop\\원드라이\\정동은\\2024\\다변량분석\\Multiple Linear Regression\\과제시작1\\data_happy_v2.CSV', index=False)


####################### 기술통계량 #######################
# -*- coding: cp949 -*-
import pandas as pd
from scipy import stats

# 데이터셋 파일 불러오기
df = pd.read_csv('C:\\Users\\User\\Desktop\\원드라이\\정동은\\2024\\다변량분석\\Multiple Linear Regression\\과제시작1\\data_happy_v1.CSV')

# 연속형 변수 선택
continuous_columns = [
    'FAM1', 'FAM15', 'AZQ1A2', 'AZQ1A4', 'AZQ2C', 'AZQ3', 'AZQ4A1', 'AZQ4A2', 'AZQ4A3',
    'AQ26A', 'AQ26A_1', 'AQ26A_2', 'AQ26A_3', 'AQ5_1', 'AQ5_2', 'AQ5_3', 'AQ5_4', 'AZQ5A1', 
    'AZQ5A2', 'INC', 'AZQ6A1', 'AZQ6A2', 'AZQ7A1', 'AZQ7A2', 'AZQ7A3', 'AZQ7A4', 'AZQ7A5', 
    'AZQ7A6', 'AZQ7A7', 'AZQ7A8', 'AZQ7A9', 'AZQ8A1', 'AZQ8A2', 'AZQ8A3', 'AZQ8A4', 'AZQ8A5', 
    'DEW3', 'DEW4'
]

# 기술 통계량 계산
description = df[continuous_columns].describe()

# 왜도(skewness), 첨도(kurtosis) 계산
skewness = df[continuous_columns].skew()
kurtosis = df[continuous_columns].kurtosis()

# Shapiro-Wilk 테스트 수행
shapiro_results = {}
for column in continuous_columns:
    shapiro_results[column] = stats.shapiro(df[column])

# 통계량과 Shapiro-Wilk 테스트 결과를 하나의 DataFrame으로 결합
stats_summary = pd.concat([description, skewness.rename('Skewness'), kurtosis.rename('Kurtosis'), pd.DataFrame(shapiro_results).T.rename(columns={0: 'Shapiro-Wilk statistic', 1: 'Shapiro-Wilk p-value'})], axis=1)

# 결과를 CSV 파일로 저장
stats_summary.to_csv('C:\\Users\\User\\Desktop\\원드라이\\정동은\\2024\\다변량분석\\Multiple Linear Regression\\과제시작1\\v2_summary_continuous.CSV')


####################### box plot #######################

import matplotlib.pyplot as plt

# 모든 열에 대한 box plot 그리기
plt.figure(figsize=(15, 10))
df.boxplot(rot=90)
plt.title('Box plot for all columns')
plt.ylabel('Values')
plt.xlabel('Columns')
plt.xticks(fontsize=8)
plt.show()

# 연속형 변수 선택
continuous_columns = [
    'FAM1', 'FAM15', 'AZQ1A2', 'AZQ1A4', 'AZQ2C', 'AZQ3', 'AZQ4A1', 'AZQ4A2', 'AZQ4A3',
    'AQ26A', 'AQ26A_1', 'AQ26A_2', 'AQ26A_3', 'AQ5_1', 'AQ5_2', 'AQ5_3', 'AQ5_4', 'AZQ5A1', 
    'AZQ5A2', 'INC', 'AZQ6A1', 'AZQ6A2', 'AZQ7A1', 'AZQ7A2', 'AZQ7A3', 'AZQ7A4', 'AZQ7A5', 
    'AZQ7A6', 'AZQ7A7', 'AZQ7A8', 'AZQ7A9', 'AZQ8A1', 'AZQ8A2', 'AZQ8A3', 'AZQ8A4', 'AZQ8A5', 
    'DEW3', 'DEW4'
]

# 연속형 변수에 대한 box plot 그리기
plt.figure(figsize=(15, 10))
df[continuous_columns].boxplot(rot=90)
plt.title('Box plot for continuous columns')
plt.ylabel('Values')
plt.xlabel('Columns')
plt.xticks(fontsize=8)
plt.show()

####################### OUTLIER 제거 #######################

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)  # 1사분위수
    Q3 = df[column].quantile(0.75)  # 3사분위수
    IQR = Q3 - Q1  # IQR (Interquartile Range)

    # 이상값의 기준 설정
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 이상값 제거
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# 연속형 변수에 대한 이상값(outlier) 제거
for column in continuous_columns:
    df = remove_outliers(df, column)

df.to_csv('C:\\Users\\User\\Desktop\\원드라이\\정동은\\2024\\다변량분석\\Multiple Linear Regression\\과제시작1\\data_happy_v3.CSV', index=False)


####################### 명목형 변수 one hot encoding #######################

# 데이터셋 파일 불러오기
df = pd.read_csv('C:\\Users\\User\\Desktop\\원드라이\\정동은\\2024\\다변량분석\\Multiple Linear Regression\\과제시작1\\data_happy_v3.csv')

# 명목형 변수 선택
nominal_columns = [
    'sq0_2', 'sq0_3', 'SQ1_1', 'SQ1_2', 'SQ1_4', 'SQ1_5', 'SQ1_6', 'SQ1_7', 'SQ1_8', 
    'AZQ2A1', 'AZQ2A2', 'AQ26', 'AQ26A_1', 'AQ26A_2', 'AQ26A_3'
]

# One-Hot Encoding 적용
df_encoded = pd.get_dummies(df, columns=nominal_columns)

# 결과 확인
print(df_encoded.head())

# 데이터셋 파일로 저장 (선택사항)
df_encoded.to_csv('C:\\Users\\User\\Desktop\\원드라이\\정동은\\2024\\다변량분석\\Multiple Linear Regression\\과제시작1\\data_happy_v3_encoded.csv', index=False)






####################### MLR 모델 학습 #######################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


from statsmodels.api import OLS

# 데이터 불러오기
df = pd.read_csv('C:\\Users\\User\\Desktop\\원드라이\\정동은\\2024\\다변량분석\\Multiple Linear Regression\\과제시작1\\data_happy_v3_encoded.csv')

# 독립 변수와 종속 변수 분리
X = df.drop(columns=['AZQ1A1'])  # 독립 변수
y = df['AZQ1A1']  # 종속 변수

# 학습 데이터와 테스트 데이터로 분할 (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 다중 선형 회귀 모델 피팅
X_train = sm.add_constant(X_train)  # 상수항 추가
model_ols = OLS(y_train, X_train)
results = model_ols.fit()

# OLS 결과창 띄우기
print(results.summary())


####################### 선형성 파악 #######################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# 데이터 불러오기
df = pd.read_csv('C:\\Users\\User\\Desktop\\원드라이\\정동은\\2024\\다변량분석\\Multiple Linear Regression\\과제시작1\\data_happy_v3_encoded.csv')

# 독립 변수와 종속 변수 분리
X = df.drop(columns=['AZQ1A1'])  # 독립 변수
y = df['AZQ1A1']  # 종속 변수

# 학습 데이터와 테스트 데이터로 분할 (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 다중 선형 회귀 모델 피팅
X_train = sm.add_constant(X_train)  # 상수항 추가
model_ols = sm.OLS(y_train, X_train)
results = model_ols.fit()

# 학습 데이터에 대한 예측값 계산
train_pred = results.predict(X_train)

# 학습 데이터에 대한 실제값과 예측값을 산점도로 표시
plt.figure(figsize=(10, 6))
plt.scatter(y_train, train_pred, color='blue', alpha=0.5)
plt.plot(y_train, y_train, color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted (Training Data)')
plt.show()

####################### 모델 성능 평가 #######################

# -*- coding: cp949 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm

# 데이터 불러오기
df = pd.read_csv('C:\\Users\\User\\Desktop\\원드라이\\정동은\\2024\\다변량분석\\Multiple Linear Regression\\과제시작1\\data_happy_v3_encoded.csv')

# MAE, MAPE, RMSE를 저장할 리스트 초기화
mae_list = []
mape_list = []
rmse_list = []

# 10번의 시행을 위한 반복문
for _ in range(100):
    # 데이터 분할
    X = df.drop(columns=['AZQ1A1'])  # 독립 변수
    y = df['AZQ1A1']  # 종속 변수
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 다중 선형 회귀 모델 피팅
    X_train = sm.add_constant(X_train)  # 상수항 추가
    model_ols = sm.OLS(y_train, X_train)
    results = model_ols.fit()
    
    # 테스트 데이터에 대한 예측값 계산
    X_test = sm.add_constant(X_test)
    test_pred = results.predict(X_test)
    
    # MAE 계산
    mae = mean_absolute_error(y_test, test_pred)
    mae_list.append(mae)
    
    # MAPE 계산
    mask = y_test != 0  # y_test가 0이 아닌 경우에 대해서만 계산
    mape = np.mean(np.abs((y_test[mask] - test_pred[mask]) / y_test[mask])) 
    mape_list.append(mape)
    
    # RMSE 계산
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    rmse_list.append(rmse)

# 결과 출력
print("MAE 평균:", np.mean(mae_list))
print("MAE 표준편차:", np.std(mae_list))
print("MAPE 평균:", np.mean(mape_list))
print("MAPE 표준편차:", np.std(mape_list))
print("RMSE 평균:", np.mean(rmse_list))
print("RMSE 표준편차:", np.std(rmse_list))


####################### (오류 수정) 명목변수 one hot encoding  #######################
import pandas as pd
# 데이터셋 파일 불러오기
df = pd.read_csv('C:\\Users\\User\\Desktop\\원드라이\\정동은\\2024\\다변량분석\\Multiple Linear Regression\\과제시작1\\data_happy_v2.CSV')

# 명목형 변수들을 one-hot encoding
nominal_columns = ['AZQ2A1', 'DEW1', 'DEW3', 'sq0_2', 'SQ1_4', 'SQ1_6', 'SQ1_7', 'SQ1_8', 'AZQ2A2', 'AQ26', 'AQ26A_1', 'AQ26A_2', 'AQ26A_3', 'DEW6']
df_encoded = pd.get_dummies(df, columns=nominal_columns)

# one-hot encoding된 데이터프레임을 CSV 파일로 저장
df_encoded.to_csv('C:\\Users\\User\\Desktop\\원드라이\\정동은\\2024\\다변량분석\\Multiple Linear Regression\\과제시작1\\data_happy_v2_encoded.csv', index=False)
