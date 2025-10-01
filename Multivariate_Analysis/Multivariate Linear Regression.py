####################### ����ġ ó�� #######################

# �����ͼ� ���� �ҷ�����
df = pd.read_csv('C:\\Users\\User\\Desktop\\�������\\������\\2024\\�ٺ����м�\\Multiple Linear Regression\\��������1\\data_happy_v1.CSV')

# AQ26A, AQ26A_1, AQ26A_2, AQ26A_3, AQ5_1, AQ5_2 ���� ����ġ�� 0���� ä���
cols_to_fill = ['AQ26A', 'AQ26A_1', 'AQ26A_2', 'AQ26A_3', 'AQ5_1', 'AQ5_2']
df[cols_to_fill] = df[cols_to_fill].fillna(0)

# �� ���� ������ ����ġ�� �ִ� �� ����
df.dropna(inplace=True)

# �����ͼ� ���Ϸ� ���� (���û���)
df.to_csv('C:\\Users\\User\\Desktop\\�������\\������\\2024\\�ٺ����м�\\Multiple Linear Regression\\��������1\\data_happy_v2.CSV', index=False)


####################### �����跮 #######################
# -*- coding: cp949 -*-
import pandas as pd
from scipy import stats

# �����ͼ� ���� �ҷ�����
df = pd.read_csv('C:\\Users\\User\\Desktop\\�������\\������\\2024\\�ٺ����м�\\Multiple Linear Regression\\��������1\\data_happy_v1.CSV')

# ������ ���� ����
continuous_columns = [
    'FAM1', 'FAM15', 'AZQ1A2', 'AZQ1A4', 'AZQ2C', 'AZQ3', 'AZQ4A1', 'AZQ4A2', 'AZQ4A3',
    'AQ26A', 'AQ26A_1', 'AQ26A_2', 'AQ26A_3', 'AQ5_1', 'AQ5_2', 'AQ5_3', 'AQ5_4', 'AZQ5A1', 
    'AZQ5A2', 'INC', 'AZQ6A1', 'AZQ6A2', 'AZQ7A1', 'AZQ7A2', 'AZQ7A3', 'AZQ7A4', 'AZQ7A5', 
    'AZQ7A6', 'AZQ7A7', 'AZQ7A8', 'AZQ7A9', 'AZQ8A1', 'AZQ8A2', 'AZQ8A3', 'AZQ8A4', 'AZQ8A5', 
    'DEW3', 'DEW4'
]

# ��� ��跮 ���
description = df[continuous_columns].describe()

# �ֵ�(skewness), ÷��(kurtosis) ���
skewness = df[continuous_columns].skew()
kurtosis = df[continuous_columns].kurtosis()

# Shapiro-Wilk �׽�Ʈ ����
shapiro_results = {}
for column in continuous_columns:
    shapiro_results[column] = stats.shapiro(df[column])

# ��跮�� Shapiro-Wilk �׽�Ʈ ����� �ϳ��� DataFrame���� ����
stats_summary = pd.concat([description, skewness.rename('Skewness'), kurtosis.rename('Kurtosis'), pd.DataFrame(shapiro_results).T.rename(columns={0: 'Shapiro-Wilk statistic', 1: 'Shapiro-Wilk p-value'})], axis=1)

# ����� CSV ���Ϸ� ����
stats_summary.to_csv('C:\\Users\\User\\Desktop\\�������\\������\\2024\\�ٺ����м�\\Multiple Linear Regression\\��������1\\v2_summary_continuous.CSV')


####################### box plot #######################

import matplotlib.pyplot as plt

# ��� ���� ���� box plot �׸���
plt.figure(figsize=(15, 10))
df.boxplot(rot=90)
plt.title('Box plot for all columns')
plt.ylabel('Values')
plt.xlabel('Columns')
plt.xticks(fontsize=8)
plt.show()

# ������ ���� ����
continuous_columns = [
    'FAM1', 'FAM15', 'AZQ1A2', 'AZQ1A4', 'AZQ2C', 'AZQ3', 'AZQ4A1', 'AZQ4A2', 'AZQ4A3',
    'AQ26A', 'AQ26A_1', 'AQ26A_2', 'AQ26A_3', 'AQ5_1', 'AQ5_2', 'AQ5_3', 'AQ5_4', 'AZQ5A1', 
    'AZQ5A2', 'INC', 'AZQ6A1', 'AZQ6A2', 'AZQ7A1', 'AZQ7A2', 'AZQ7A3', 'AZQ7A4', 'AZQ7A5', 
    'AZQ7A6', 'AZQ7A7', 'AZQ7A8', 'AZQ7A9', 'AZQ8A1', 'AZQ8A2', 'AZQ8A3', 'AZQ8A4', 'AZQ8A5', 
    'DEW3', 'DEW4'
]

# ������ ������ ���� box plot �׸���
plt.figure(figsize=(15, 10))
df[continuous_columns].boxplot(rot=90)
plt.title('Box plot for continuous columns')
plt.ylabel('Values')
plt.xlabel('Columns')
plt.xticks(fontsize=8)
plt.show()

####################### OUTLIER ���� #######################

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)  # 1�������
    Q3 = df[column].quantile(0.75)  # 3�������
    IQR = Q3 - Q1  # IQR (Interquartile Range)

    # �̻��� ���� ����
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # �̻� ����
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# ������ ������ ���� �̻�(outlier) ����
for column in continuous_columns:
    df = remove_outliers(df, column)

df.to_csv('C:\\Users\\User\\Desktop\\�������\\������\\2024\\�ٺ����м�\\Multiple Linear Regression\\��������1\\data_happy_v3.CSV', index=False)


####################### ����� ���� one hot encoding #######################

# �����ͼ� ���� �ҷ�����
df = pd.read_csv('C:\\Users\\User\\Desktop\\�������\\������\\2024\\�ٺ����м�\\Multiple Linear Regression\\��������1\\data_happy_v3.csv')

# ����� ���� ����
nominal_columns = [
    'sq0_2', 'sq0_3', 'SQ1_1', 'SQ1_2', 'SQ1_4', 'SQ1_5', 'SQ1_6', 'SQ1_7', 'SQ1_8', 
    'AZQ2A1', 'AZQ2A2', 'AQ26', 'AQ26A_1', 'AQ26A_2', 'AQ26A_3'
]

# One-Hot Encoding ����
df_encoded = pd.get_dummies(df, columns=nominal_columns)

# ��� Ȯ��
print(df_encoded.head())

# �����ͼ� ���Ϸ� ���� (���û���)
df_encoded.to_csv('C:\\Users\\User\\Desktop\\�������\\������\\2024\\�ٺ����м�\\Multiple Linear Regression\\��������1\\data_happy_v3_encoded.csv', index=False)






####################### MLR �� �н� #######################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


from statsmodels.api import OLS

# ������ �ҷ�����
df = pd.read_csv('C:\\Users\\User\\Desktop\\�������\\������\\2024\\�ٺ����м�\\Multiple Linear Regression\\��������1\\data_happy_v3_encoded.csv')

# ���� ������ ���� ���� �и�
X = df.drop(columns=['AZQ1A1'])  # ���� ����
y = df['AZQ1A1']  # ���� ����

# �н� �����Ϳ� �׽�Ʈ �����ͷ� ���� (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ���� ���� ȸ�� �� ����
X_train = sm.add_constant(X_train)  # ����� �߰�
model_ols = OLS(y_train, X_train)
results = model_ols.fit()

# OLS ���â ����
print(results.summary())


####################### ������ �ľ� #######################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# ������ �ҷ�����
df = pd.read_csv('C:\\Users\\User\\Desktop\\�������\\������\\2024\\�ٺ����м�\\Multiple Linear Regression\\��������1\\data_happy_v3_encoded.csv')

# ���� ������ ���� ���� �и�
X = df.drop(columns=['AZQ1A1'])  # ���� ����
y = df['AZQ1A1']  # ���� ����

# �н� �����Ϳ� �׽�Ʈ �����ͷ� ���� (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ���� ���� ȸ�� �� ����
X_train = sm.add_constant(X_train)  # ����� �߰�
model_ols = sm.OLS(y_train, X_train)
results = model_ols.fit()

# �н� �����Ϳ� ���� ������ ���
train_pred = results.predict(X_train)

# �н� �����Ϳ� ���� �������� �������� �������� ǥ��
plt.figure(figsize=(10, 6))
plt.scatter(y_train, train_pred, color='blue', alpha=0.5)
plt.plot(y_train, y_train, color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted (Training Data)')
plt.show()

####################### �� ���� �� #######################

# -*- coding: cp949 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm

# ������ �ҷ�����
df = pd.read_csv('C:\\Users\\User\\Desktop\\�������\\������\\2024\\�ٺ����м�\\Multiple Linear Regression\\��������1\\data_happy_v3_encoded.csv')

# MAE, MAPE, RMSE�� ������ ����Ʈ �ʱ�ȭ
mae_list = []
mape_list = []
rmse_list = []

# 10���� ������ ���� �ݺ���
for _ in range(100):
    # ������ ����
    X = df.drop(columns=['AZQ1A1'])  # ���� ����
    y = df['AZQ1A1']  # ���� ����
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # ���� ���� ȸ�� �� ����
    X_train = sm.add_constant(X_train)  # ����� �߰�
    model_ols = sm.OLS(y_train, X_train)
    results = model_ols.fit()
    
    # �׽�Ʈ �����Ϳ� ���� ������ ���
    X_test = sm.add_constant(X_test)
    test_pred = results.predict(X_test)
    
    # MAE ���
    mae = mean_absolute_error(y_test, test_pred)
    mae_list.append(mae)
    
    # MAPE ���
    mask = y_test != 0  # y_test�� 0�� �ƴ� ��쿡 ���ؼ��� ���
    mape = np.mean(np.abs((y_test[mask] - test_pred[mask]) / y_test[mask])) 
    mape_list.append(mape)
    
    # RMSE ���
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    rmse_list.append(rmse)

# ��� ���
print("MAE ���:", np.mean(mae_list))
print("MAE ǥ������:", np.std(mae_list))
print("MAPE ���:", np.mean(mape_list))
print("MAPE ǥ������:", np.std(mape_list))
print("RMSE ���:", np.mean(rmse_list))
print("RMSE ǥ������:", np.std(rmse_list))


####################### (���� ����) ��񺯼� one hot encoding  #######################
import pandas as pd
# �����ͼ� ���� �ҷ�����
df = pd.read_csv('C:\\Users\\User\\Desktop\\�������\\������\\2024\\�ٺ����м�\\Multiple Linear Regression\\��������1\\data_happy_v2.CSV')

# ����� �������� one-hot encoding
nominal_columns = ['AZQ2A1', 'DEW1', 'DEW3', 'sq0_2', 'SQ1_4', 'SQ1_6', 'SQ1_7', 'SQ1_8', 'AZQ2A2', 'AQ26', 'AQ26A_1', 'AQ26A_2', 'AQ26A_3', 'DEW6']
df_encoded = pd.get_dummies(df, columns=nominal_columns)

# one-hot encoding�� �������������� CSV ���Ϸ� ����
df_encoded.to_csv('C:\\Users\\User\\Desktop\\�������\\������\\2024\\�ٺ����м�\\Multiple Linear Regression\\��������1\\data_happy_v2_encoded.csv', index=False)