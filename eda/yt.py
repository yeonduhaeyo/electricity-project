import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy
from pathlib import Path

# 한글 폰트 설정 (Windows 기본 값: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 경로
BASE_DIR = Path(__file__).resolve().parents[0]
TRAIN_DATA_PATH = BASE_DIR / 'data' / 'train.csv'
TEST_DATA_PATH = BASE_DIR / 'data' / 'test.csv'
WEATHER_DATA_PATH = BASE_DIR / 'data' / '청주_기상_2024년도.csv'

# 데이터 로드
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)
weather_df = pd.read_csv(WEATHER_DATA_PATH, encoding='cp949')

# 데이터 기본 정보
train_df.info()
train_df.isna().sum()
train_df.head()

weather_df.info()
weather_df.isna().sum()
weather_df.head()

# 범주형 컬럼 확인
train_df["작업유형"].unique()
train_df["작업유형"].value_counts()

# 측정일시 데이터 타입 변환 및 파생변수 생성
train_df["측정일시"] = pd.to_datetime(train_df["측정일시"], format="%Y-%m-%d %H:%M:%S")
train_df["year"] = train_df["측정일시"].dt.year
train_df["month"] = train_df["측정일시"].dt.month
train_df["day"] = train_df["측정일시"].dt.day
train_df["hour"] = train_df["측정일시"].dt.hour
train_df["minute"] = train_df["측정일시"].dt.minute
train_df["second"] = train_df["측정일시"].dt.second
train_df["weekday"] = train_df["측정일시"].dt.weekday
train_df["is_weekend"] = train_df["weekday"].isin([5,6]).astype(int)

# 2024년 연휴 여부 컬럼 생성
holiday_periods = [
	("2024-01-01", "2024-01-01"),
	("2024-02-09", "2024-02-12"),
	("2024-03-01", "2024-03-01"),
	("2024-04-10", "2024-04-10"),
	("2024-05-05", "2024-05-05"),
	("2024-05-06", "2024-05-06"),
	("2024-05-15", "2024-05-15"),
	("2024-06-06", "2024-06-06"),
	("2024-08-15", "2024-08-15"),
	("2024-09-17", "2024-09-19"),
	("2024-10-03", "2024-10-03"),
	("2024-10-09", "2024-10-09"),
	("2024-12-25", "2024-12-25"),
]

holiday_dates = set()
for start_date, end_date in holiday_periods:
	holiday_dates.update(pd.date_range(start=start_date, end=end_date, freq="D").date)

train_df["is_holiday"] = train_df["측정일시"].dt.date.apply(lambda d: int(d in holiday_dates))

# year, second 컬럼 제거 (단일값)
train_df["year"].nunique()
train_df["second"].nunique()
train_df.drop(columns=["year", "second"], inplace=True)

# 파생변수 생성

# 학습 데이터에 기상 데이터 병합
weather_df.drop(columns=["지점", "지점명"], inplace=True)
weather_df["일시"] = pd.to_datetime(weather_df["일시"], format="%Y-%m-%d %H:%M")

train_df["측정일시_분"] = train_df["측정일시"].dt.floor("H")
train_df = pd.merge(train_df, weather_df, left_on="측정일시_분", right_on="일시", how="left")

# 결측치 재확인
train_df.isna().sum()
# 강수량 결측치는 0으로 대체
train_df["강수량(mm)"].fillna(0, inplace=True)

# 기상 데이터에서 누락 (9월19일, 9월20일)
train_df.loc[train_df.isna().sum(axis=1)>0, :]