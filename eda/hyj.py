import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/train.csv")
df['측정일시'] = pd.to_datetime(df['측정일시'])

df.head()
df.tail()

df.info()
df.isna().sum() # 결측치 없음

df.describe() # 기초 통계

### 시계열 기준 데이터 시각화

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

df = df.sort_values('측정일시')

cols = [
    '전력사용량(kWh)',
    '지상무효전력량(kVarh)',
    '진상무효전력량(kVarh)',
    '탄소배출량(tCO2)',
    '지상역률(%)',
    '진상역률(%)'
]

n = len(cols)
plt.figure(figsize=(14, 3 * n))  # 변수 개수에 따라 높이 자동 조절

for i, col in enumerate(cols, start=1):
    plt.subplot(n, 1, i)
    plt.plot(df['측정일시'], df[col], label=col, color='tab:blue')
    plt.title(col)
    plt.xlabel('시간')
    plt.ylabel('값')
    plt.grid(True)
    plt.tight_layout()

plt.show()


### 범주형 변수 확인
df["작업유형"].unique()
df["작업유형"].value_counts()
# Light_Load
# Medium_Load
# Maximum_Load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 0) 로드 & 기본 확인 =====
df = pd.read_csv("./data/train.csv")
df['측정일시'] = pd.to_datetime(df['측정일시'], errors='coerce')

# === 자정(00:00) 오표기 보정: 00:00 이면서 직전 행과 '같은 날짜'로 찍힌 경우만 +1일 ===
midnight = (df['측정일시'].dt.hour == 0) & (df['측정일시'].dt.minute == 0)
same_day_as_prev = df['측정일시'].dt.normalize().eq(df['측정일시'].shift(1).dt.normalize())
bug_mask = midnight & same_day_as_prev
df.loc[bug_mask, '측정일시'] = df.loc[bug_mask, '측정일시'] + pd.Timedelta(days=1)

# 정렬
df = df.sort_values('측정일시').reset_index(drop=True)

# ===== 기본 출력 =====
print("=== head ==="); print(df.head(), "\n")
print("=== tail ==="); print(df.tail(), "\n")
print("=== info ==="); print(df.info(), "\n")
print("=== 결측치 합계 ==="); print(df.isna().sum(), "\n")
print("=== 기술통계(수치) ===")
print(df.select_dtypes(include=[np.number]).describe(percentiles=[.01,.25,.5,.75,.99]).T, "\n")

# ===== 1) 15분 간격/결측/중복 체크 =====
df_idx = df.set_index('측정일시')
full_idx = pd.date_range(df_idx.index.min(), df_idx.index.max(), freq='15T')

missing_times = full_idx.difference(df_idx.index)
dup_count = df_idx.index.duplicated().sum()

print(f"[간격체크] 빠진 타임스탬프 수: {len(missing_times)}")
print(f"[간격체크] 중복 타임스탬프 수: {dup_count}\n")

# ===== 2) 시계열 시각화(핵심 변수) =====
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

cols = [
    '전력사용량(kWh)',
    '지상무효전력량(kVarh)',
    '진상무효전력량(kVarh)',
    '탄소배출량(tCO2)',
    '지상역률(%)',
    '진상역률(%)',
    '전기요금(원)'
]
cols_plot = [c for c in cols if c in df.columns]

n = len(cols_plot)
if n > 0:
    plt.figure(figsize=(14, 2.6 * n))
    for i, col in enumerate(cols_plot, start=1):
        plt.subplot(n, 1, i)
        plt.plot(df['측정일시'], df[col], label=col)
        plt.title(col); plt.xlabel('시간'); plt.ylabel('값'); plt.grid(True)
    plt.tight_layout(); plt.show()

# ===== 3) 범주형(작업유형) 확인 =====
if '작업유형' in df.columns:
    print("=== 작업유형 .unique() ==="); print(df["작업유형"].unique(), "\n")
    print("=== 작업유형 .value_counts() ==="); print(df["작업유형"].value_counts(), "\n")
else:
    print("[알림] '작업유형' 컬럼이 없습니다.\n")

# ===== 4) 캘린더 피처(테스트에서도 만들 수 있는 것만) =====
df['hour'] = df['측정일시'].dt.hour
df['dow'] = df['측정일시'].dt.dayofweek      # 월=0 … 일=6
df['month'] = df['측정일시'].dt.month
df['is_weekend'] = (df['dow'] >= 5).astype(int)

def season(m):
    return 'summer' if m in [6,7,8] else ('winter' if m in [11,12,1,2] else 'spring_fall')
df['season'] = df['month'].map(season)

# ===== 5) 파생 지표(요금 분해에 꼭 필요한 3가지) =====
# (1) 단가(원/kWh) = 전기요금 / 전력사용량
safe = ('전력사용량(kWh)' in df.columns) & ('전기요금(원)' in df.columns)
if safe:
    mask = df['전력사용량(kWh)'] > 1e-6
    df['단가(원/kWh)'] = np.where(mask, df['전기요금(원)'] / df['전력사용량(kWh)'], np.nan)

# (2) 무효합 & 무효비 = (지상+진상)/사용량
if {'지상무효전력량(kVarh)','진상무효전력량(kVarh)'}.issubset(df.columns):
    df['무효합(kVarh)'] = df['지상무효전력량(kVarh)'] + df['진상무효전력량(kVarh)']
    if '전력사용량(kWh)' in df.columns:
        mask = df['전력사용량(kWh)'] > 1e-6
        df['무효비'] = np.where(mask, df['무효합(kVarh)'] / df['전력사용량(kWh)'], np.nan)

# (3) PF(%) = 지상/진상 중 큰 값(효율감)
if {'지상역률(%)','진상역률(%)'}.issubset(df.columns):
    df['PF(%)'] = df[['지상역률(%)','진상역률(%)']].max(axis=1)

# 빠르게 시계열 확인
quick_cols = ['전기요금(원)','전력사용량(kWh)','단가(원/kWh)','무효비','PF(%)']
quick_cols = [c for c in quick_cols if c in df.columns]

if len(quick_cols) > 0:
    plt.figure(figsize=(14, 2.6 * len(quick_cols)))
    for i, col in enumerate(quick_cols, start=1):
        plt.subplot(len(quick_cols), 1, i)
        plt.plot(df['측정일시'], df[col])
        plt.title(f"{col} (시계열)"); plt.xlabel('시간'); plt.ylabel(col); plt.grid(True)
    plt.tight_layout(); plt.show()

# ===== 6) 시간/요일/월/작업유형별 평균 요금 =====
def bar_mean(by, target='전기요금(원)'):
    base_cols = [c for c in [target,'전력사용량(kWh)','단가(원/kWh)','무효비','PF(%)'] if c in df.columns]
    if (by not in df.columns) or (target not in df.columns) or len(base_cols) == 0:
        return None
    g = df.groupby(by)[base_cols].mean()
    ax = g[target].plot(kind='bar', rot=0, figsize=(10,3), title=f'{by}별 평균 {target}')
    ax.set_xlabel(by); ax.set_ylabel('평균 요금(원)'); plt.tight_layout(); plt.show()
    return g

g_hour = bar_mean('hour')
g_dow  = bar_mean('dow')
g_mon  = bar_mean('month')

# 작업유형 상위만 보기(많을 수 있어서)
if '작업유형' in df.columns and '전기요금(원)' in df.columns:
    top_types = df['작업유형'].value_counts().head(10).index
    g_type = df[df['작업유형'].isin(top_types)].groupby('작업유형')[[c for c in ['전기요금(원)','전력사용량(kWh)'] if c in df.columns]] \
             .mean().sort_values('전기요금(원)')
    ax = g_type['전기요금(원)'].plot(kind='barh', figsize=(8,4), title='작업유형별 평균 요금(상위 10개)')
    ax.set_xlabel('평균 요금(원)'); ax.set_ylabel('작업유형'); plt.tight_layout(); plt.show()

# ===== 7) 피크 구간(상위 5%) — 비용 리스크 확인 =====
if '전기요금(원)' in df.columns:
    q95 = df['전기요금(원)'].quantile(0.95)
    peak = df[df['전기요금(원)'] >= q95]
    print(f"[피크] 상위 5% 개수: {len(peak)}")
    if len(peak) > 0:
        peak['hour'].value_counts().sort_index().plot(kind='bar', figsize=(10,3), title='피크(상위 5%) 발생 시간대')
        plt.xlabel('hour'); plt.ylabel('count'); plt.tight_layout(); plt.show()
        if '작업유형' in peak.columns:
            print("[피크 시간대 작업유형 Top5]\n", peak['작업유형'].value_counts().head(5), "\n")

# ===== 8) 간단 상관 — 타깃과 주요 수치 변수 (스피어만) =====
num_for_corr = [c for c in [
    '전력사용량(kWh)','지상무효전력량(kVarh)','진상무효전력량(kVarh)',
    '탄소배출량(tCO2)','지상역률(%)','진상역률(%)',
    '무효합(kVarh)','무효비','단가(원/kWh)','전기요금(원)'
] if c in df.columns]

if '전기요금(원)' in num_for_corr:
    corr = df[num_for_corr].corr(method='spearman')['전기요금(원)'].sort_values(ascending=False)
    print("=== Spearman 상관(타깃 vs 변수) ==="); print(corr, "\n")
    corr.drop('전기요금(원)').plot(kind='bar', figsize=(10,3), title='Spearman 상관 (전기요금 vs 변수)')
    plt.ylabel('상관계수'); plt.tight_layout(); plt.show()
else:
    print("[알림] 상관 분석을 위한 '전기요금(원)' 컬럼이 없습니다.\n")

print("\n[완료] EDA 핵심 체크 끝!")
print("- 00:00 오표기 보정 적용")
print("- 간격/결측/중복 결과 확인")
print("- 단가/무효비/PF로 요금 분해 감 잡기")
print("- 시간/요일/월/작업유형 패턴 파악")
print("- 피크(상위5%) 시간대·작업유형 확인")
print("- 상관으로 중요 신호 스크린")