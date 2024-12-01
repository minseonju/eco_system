
import pandas as pd

# CSV 파일을 읽어들입니다.
df = pd.read_csv('CO2 Emissions_Canada.csv')

# 데이터 상위 5개 행을 확인합니다.
df.head()

print(df.columns)


# 데이터 상위 5개 행을 확인합니다.
df.head()

# 데이터 그룹화 (예: 연료 종류별로 그룹화하여 평균, 최댓값, 최솟값 계산)
fuel_group = df.groupby('Fuel Type').agg({
    'CO2 Emissions(g/km)': ['mean', 'max', 'min'],
    'Engine Size(L)': ['mean', 'max', 'min'],
    'Cylinders': ['mean', 'max', 'min']
})

print(fuel_group)


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
