
from pathlib import Path
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from scipy.stats import loguniform, randint
from sklearn.base import clone

# -----------------------
# 경로/설정
# -----------------------
BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE_DIR / 'data' / 'processed' / 'train_yt.csv'
TEST_PATH  = BASE_DIR / 'data' / 'test.csv'
SUB_PATH   = BASE_DIR / 'data' / 'output' / 'submission.csv'
RANDOM_STATE = 42
N_SPLITS = 5
N_ITER_STUDENT = 60   # 학생 튜닝 예산
ALPHA_CANDIDATES = [0.7, 0.8, 0.9]  # 증류 비중 후보

# -----------------------
# 2024 공휴일
# -----------------------
def get_korean_holidays_2024():
    days = set()
    def add(d): days.add(pd.to_datetime(d).date())
    def add_range(s,e):
        for d in pd.date_range(s,e,freq="D"):
            days.add(d.date())
    add("2024-01-01")  # 신정
    add_range("2024-02-09","2024-02-12")  # 설날 연휴
    add("2024-03-01")  # 3·1절
    add("2024-04-10")  # 국회의원 선거
    add("2024-05-05")  # 어린이날
    add("2024-05-06")  # 대체공휴일
    add("2024-05-15")  # 부처님오신날
    add("2024-06-06")  # 현충일
    add("2024-08-15")  # 광복절
    add_range("2024-09-17","2024-09-19")  # 추석 연휴
    add("2024-10-03")  # 개천절
    add("2024-10-09")  # 한글날
    add("2024-12-25")  # 크리스마스
    return days

HOLIDAYS_2024 = get_korean_holidays_2024()

# -----------------------
# 계절 매핑
# -----------------------
season_mapping = {
    11: 'winter', 12: 'winter', 1: 'winter',
    2: 'spring', 3: 'spring', 4: 'spring',
    5: 'summer', 6: 'summer', 7: 'summer', 8: 'summer',
    9: 'autumn', 10: 'autumn'
}
SEASON_ORDER = ['winter', 'spring', 'summer', 'autumn']

def add_season_feature(df: pd.DataFrame, dt_col: str = "측정일시", col_name: str = "season") -> pd.DataFrame:
    """
    dt_col에서 month를 추출해 season_mapping으로 매핑한 'season' 컬럼을 추가합니다.
    - train은 건드리지 않고, test 전용 메타 컬럼으로만 사용.
    - category 타입과 고정 카테고리 순서를 설정(일관성을 위해).
    """
    out = df.copy()
    month = pd.to_datetime(out[dt_col]).dt.month
    season = month.map(season_mapping)
    out[col_name] = pd.Categorical(season, categories=SEASON_ORDER, ordered=True)
    return out

# -----------------------
# 테스트 피처 빌드 함수
# -----------------------
def build_test_features(df, dt_col="측정일시"):
    df = df.copy()
    dt = pd.to_datetime(df[dt_col])
    df["month"]   = dt.dt.month.astype("int16")
    df["day"]     = dt.dt.day.astype("int16")
    df["hour"]    = dt.dt.hour.astype("int16")
    df["minute"]  = dt.dt.minute.astype("int16")
    df["weekday"] = dt.dt.weekday.astype("int16")
    df["is_weekend"] = (df["weekday"] >= 5).astype("int8")
    df["is_holiday"] = dt.dt.date.map(lambda d: 1 if d in HOLIDAYS_2024 else 0).astype("int8")
    df["is_12"] = (df["hour"] == 12).astype("int8")
    df["is_21"] = (df["hour"] == 21).astype("int8")
    df = add_season_feature(df, dt_col=dt_col, col_name="season")
    return df

# -----------------------
# 데이터 로드
# -----------------------
train = pd.read_csv(TRAIN_PATH, parse_dates=["측정일시"])
test  = pd.read_csv(TEST_PATH,  parse_dates=["측정일시"])

TARGET = "전기요금(원)"

# -----------------------
# Teacher 설정
#  - 가능한 많은 유용 피처 사용 (train에만 존재)
#  - id/측정일시/타깃/작업유형/season 제외한 수치형 + 작업유형(범주형) 모두 사용
# -----------------------
all_cols = train.columns.tolist()
teacher_num = [
    c for c in all_cols
    if c not in ["id","측정일시","작업유형", "season", TARGET] and train[c].dtype != 'O'
]
teacher_cat = ["작업유형", "season"]  # 범주형

teacher_X = train[teacher_cat + teacher_num]
teacher_y = train[TARGET].astype(float)

# 전처리 & 모델 (Teacher)
teacher_pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), teacher_cat),
        ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median")),
                               ("sc", StandardScaler())]), teacher_num),
    ],
    remainder="drop",
)

teacher_model = HistGradientBoostingRegressor(
    random_state=RANDOM_STATE,
    early_stopping=True,
    validation_fraction=0.1,
    max_iter=800,
)

teacher_pipe = Pipeline([("prep", teacher_pre), ("model", teacher_model)])

# -----------------------
# Teacher OOF 예측
# OOF란 교차검증 과정에서 각 fold의 검증 세트에 대한 예측을 모은 것
# -----------------------
sorted_idx = train["측정일시"].argsort(kind="mergesort").values
teacher_X_sorted = teacher_X.iloc[sorted_idx].reset_index(drop=True)
teacher_y_sorted = teacher_y.iloc[sorted_idx].reset_index(drop=True)

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
oof_pred = np.zeros(len(teacher_X_sorted), dtype=float)

for tr_idx, va_idx in tscv.split(teacher_X_sorted):
    X_tr, X_va = teacher_X_sorted.iloc[tr_idx], teacher_X_sorted.iloc[va_idx]
    y_tr, y_va = teacher_y_sorted.iloc[tr_idx], teacher_y_sorted.iloc[va_idx]
    pipe_fold = clone(teacher_pipe)
    pipe_fold.fit(X_tr, y_tr)
    oof_pred[va_idx] = pipe_fold.predict(X_va)

teacher_oof_mse = mean_squared_error(teacher_y_sorted, oof_pred)
print(f"[Teacher] OOF MSE: {teacher_oof_mse:.6f}")

# -----------------------
# Student 설정
# -----------------------
student_num = ["month","day","hour","minute","weekday","is_weekend","is_holiday", "is_12", "is_21"]
student_cat = ["작업유형", "season"]

student_train_X = train[student_cat + student_num].iloc[sorted_idx].reset_index(drop=True)
student_train_y = teacher_y_sorted  # 원 타깃 정렬

# test 파생 생성
test_feat = build_test_features(test, "측정일시")
student_test_X = pd.concat(
    [test[["작업유형"]].reset_index(drop=True),
     test_feat[student_num].reset_index(drop=True),
     test_feat["season"].reset_index(drop=True)],
    axis=1
)

# -----------------------
# Alpha 선택: y_blend = (1-α)*y + α*teacher_oof
# 마지막 split을 홀드아웃으로 사용해 α 선택
# -----------------------
split_list = list(tscv.split(student_train_X))
last_tr_idx, last_va_idx = split_list[-1]
best_alpha, best_mse = None, float("inf")

for alpha in ALPHA_CANDIDATES:
    y_blend = (1 - alpha) * student_train_y + alpha * oof_pred
    # 간단한 학생 베이스(튜닝 전)으로 홀드아웃 평가
    base_pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), student_cat),
            ("num", StandardScaler(), student_num),
        ],
        remainder="drop",
    )
    base_model = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        max_iter=600,
    )
    base_pipe = Pipeline([("prep", base_pre), ("model", base_model)])
    base_pipe.fit(student_train_X.iloc[last_tr_idx], y_blend.iloc[last_tr_idx] if isinstance(y_blend, pd.Series) else y_blend[last_tr_idx])
    va_pred = base_pipe.predict(student_train_X.iloc[last_va_idx])
    mse = mean_squared_error(student_train_y.iloc[last_va_idx], va_pred)  # 진짜 y로 평가
    print(f"[Alpha Search] alpha={alpha} -> Holdout MSE={mse:.6f}")
    if mse < best_mse:
        best_mse, best_alpha = mse, alpha

print(f"[Alpha Search] Best alpha={best_alpha} (Holdout MSE={best_mse:.6f})")

# 최종 증류 타깃 생성
y_distill = (1 - best_alpha) * student_train_y + best_alpha * oof_pred

# -----------------------
# Student 튜닝 (RandomizedSearchCV, MSE)
# -----------------------
student_pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), student_cat),
        ("num", StandardScaler(), student_num),
    ],
    remainder="drop",
)

student_model = HistGradientBoostingRegressor(
    random_state=RANDOM_STATE,
    early_stopping=True,
    validation_fraction=0.1,
)

student_pipe = Pipeline([("prep", student_pre), ("model", student_model)])

param_distributions = {
    "model__learning_rate": loguniform(1e-3, 3e-1),
    "model__max_iter": randint(300, 1200),
    "model__max_depth": randint(3, 20),
    "model__max_leaf_nodes": randint(31, 255),
    "model__min_samples_leaf": randint(20, 300),
    "model__l2_regularization": loguniform(1e-8, 1e-1),
    "model__max_bins": randint(64, 255),
}

search = RandomizedSearchCV(
    estimator=student_pipe,
    param_distributions=param_distributions,
    n_iter=N_ITER_STUDENT,
    scoring="neg_mean_squared_error",
    cv=tscv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
    refit=True,
)

search.fit(student_train_X, y_distill)
best_cv_mse = -search.best_score_
print(f"\n[Student] Best CV MSE (distilled target): {best_cv_mse:.6f}")
print("Best Params:")
for k,v in search.best_params_.items():
    print(f"  {k}: {v}")

# 실제 타깃 기준 홀드아웃 MSE도 체크
best_student = search.best_estimator_
best_student.fit(student_train_X.iloc[last_tr_idx], y_distill.iloc[last_tr_idx] if isinstance(y_distill, pd.Series) else y_distill[last_tr_idx])
holdout_pred = best_student.predict(student_train_X.iloc[last_va_idx])
holdout_mse_true = mean_squared_error(student_train_y.iloc[last_va_idx], holdout_pred)
print(f"[Student] Holdout MSE vs TRUE y: {holdout_mse_true:.6f}")

# -----------------------
# Test 예측 & 저장
# -----------------------
final_model = search.best_estimator_
final_model.fit(student_train_X, y_distill)
test_pred = final_model.predict(student_test_X)
test_pred = np.clip(test_pred, 0, None)

if "id" in test.columns:
    sub = pd.DataFrame({"id": test["id"], "전기요금(원)": test_pred})
else:
    sub = pd.DataFrame({"id": np.arange(len(test_pred)), "전기요금(원)": test_pred})

sub.to_csv(SUB_PATH, index=False, encoding="utf-8-sig")
print(f"Saved: {SUB_PATH}")
