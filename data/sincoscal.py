import pandas as pd
import numpy as np

# 1. 1년치 일별 데이터 생성 (365일)
dates = pd.date_range(start='2025-09-01', end='2026-12-31', freq='D')
df = pd.DataFrame({'date': dates})
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# 2. 일(Day) 변수를 1년 주기(365.25일) 기준 순환형 sin/cos 인코딩으로 변환
df['day_of_year'] = df['date'].dt.dayofyear
df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

# 3. CSV 파일로 저장
df.to_csv('yearly_day_encoding.csv', index=False, encoding='utf-8-sig')
