#데이터 그룹화
import pandas as pd

# CSV 파일을 읽기
df = pd.read_csv('CO2 Emissions_Canada.csv')

# 데이터 상위 5개 행을 확인
df.head()

print(df.columns)


# 데이터 상위 5개 행 확인
df.head()

# 데이터 그룹화 (예: 연료 종류별로 그룹화하여 평균, 최댓값, 최솟값 계산)
fuel_group = df.groupby('Fuel Type').agg({
    'CO2 Emissions(g/km)': ['mean', 'max', 'min'],
    'Engine Size(L)': ['mean', 'max', 'min'],
    'Cylinders': ['mean', 'max', 'min']
})

print(fuel_group)


# 그래프 그리기
import matplotlib.pyplot as plt
import seaborn as sns

# 1) 연료 종류별 이산화탄소 배출량 분포 시각화
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Fuel Type', y='CO2 Emissions(g/km)', palette='Set2')
plt.title('CO2 Emissions by Fuel Type')
plt.xticks(rotation=45)
plt.show()

# 2) 엔진 크기와 이산화탄소 배출량의 관계 시각화
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Engine Size(L)', y='CO2 Emissions(g/km)', hue='Fuel Type')
plt.title('CO2 Emissions vs Engine Size')
plt.show()


#머신러닝
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 'CO2 Emissions(g/km)'을 예측할 대상으로 설정
X = df[['Engine Size(L)', 'Cylinders', 'Fuel Type']]
y = df['CO2 Emissions(g/km)']

# 범주형 변수 처리: Fuel Type을 숫자로 변환 
X = pd.get_dummies(X, drop_first=True)

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 모델 평가
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")
