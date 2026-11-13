import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="시도별 교통사고 건수 예측",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 시도별 전년 대비 교통사고 건수 예측")
st.caption("2021~2024년 한국도로교통공단 데이터를 이용해 2025년 시도별 교통사고 건수 변화를 예측합니다.")

# CSV 3개의 실제 데이터를 코드 안에 포함했습니다.
# 연도별 값은 각 CSV의 '시도', 연도별 사고 건수 열에서 가져왔습니다.
ANNUAL_DATA = {2021: {'서울': 33913, '부산': 11339, '대구': 12133, '인천': 7912, '광주': 7543, '대전': 7027, '울산': 3801, '세종': 788, '경기': 53332, '강원': 6605, '충북': 7985, '충남': 8423, '전북': 6112, '전남': 8685, '경북': 12100, '경남': 11059, '제주': 4373}, 2022: {'서울': 33698, '부산': 10882, '대구': 11102, '인천': 7695, '광주': 7122, '대전': 6841, '울산': 3660, '세종': 932, '경기': 52968, '강원': 6425, '충북': 7644, '충남': 8164, '전북': 5681, '전남': 8335, '경북': 11037, '경남': 10611, '제주': 4039}, 2023: {'서울': 33811, '부산': 11003, '경기': 53164, '강원': 6079, '충북': 7939, '충남': 8606, '전북': 5830, '전남': 8287, '경북': 10777, '경남': 11122, '제주': 3950, '대구': 10880, '인천': 7648, '광주': 6972, '대전': 7398, '울산': 3664, '세종': 1166}, 2024: {'서울': 33465, '부산': 11255, '경기': 52175, '강원': 5930, '충북': 7989, '충남': 8896, '전북': 5841, '전남': 8213, '경북': 10403, '경남': 10703, '제주': 3963, '대구': 10222, '인천': 8089, '광주': 6908, '대전': 7490, '울산': 3576, '세종': 1231}}

# -------------------------------------------------------------------
# 1. 데이터 정리
# -------------------------------------------------------------------
records = []
for year, region_counts in ANNUAL_DATA.items():
    for region, count in region_counts.items():
        records.append({
            "시도": region,
            "연도": int(year),
            "사고건수": int(count)
        })

long_df = pd.DataFrame(records)
wide_df = (
    long_df.pivot(index="시도", columns="연도", values="사고건수")
    .reset_index()
)
wide_df.columns.name = None

required_years = [2021, 2022, 2023, 2024]
wide_df = wide_df.dropna(subset=required_years).copy()
for year in required_years:
    wide_df[year] = wide_df[year].astype(int)

# 학습용 자료: 기준연도 사고건수로 다음 해의 증감량을 예측
model_rows = []
for _, row in wide_df.iterrows():
    region = row["시도"]
    for base_year in [2021, 2022, 2023]:
        base_count = int(row[base_year])
        next_count = int(row[base_year + 1])
        previous_change = (
            int(row[base_year]) - int(row[base_year - 1])
            if base_year > 2021 else np.nan
        )
        previous_rate = (
            previous_change / int(row[base_year - 1]) * 100
            if base_year > 2021 else np.nan
        )

        model_rows.append({
            "시도": region,
            "기준연도": base_year,
            "기준연도 사고건수": base_count,
            "직전 연도 증감": previous_change,
            "직전 연도 증감률": previous_rate,
            "실제 전년 대비 증감": next_count - base_count,
            "예측 대상 연도": base_year + 1
        })

model_df = pd.DataFrame(model_rows).sort_values(["시도", "기준연도"])

# 2025년 예측용 입력 자료
future_rows = []
for _, row in wide_df.iterrows():
    previous_change = int(row[2024]) - int(row[2023])
    previous_rate = previous_change / int(row[2023]) * 100
    future_rows.append({
        "시도": row["시도"],
        "기준연도": 2024,
        "기준연도 사고건수": int(row[2024]),
        "직전 연도 증감": previous_change,
        "직전 연도 증감률": previous_rate
    })
future_df = pd.DataFrame(future_rows).sort_values("시도")

# -------------------------------------------------------------------
# 2. 데이터 개요
# -------------------------------------------------------------------
st.subheader("1. 데이터 확인")
c1, c2, c3 = st.columns(3)
c1.metric("분석 시도 수", f"{len(wide_df)}개")
c2.metric("사용 연도", "2021~2024년")
c3.metric("학습 데이터 행 수", f"{len(model_df)}개")

with st.expander("연도별 원본 사고 건수 보기"):
    display_df = wide_df[["시도"] + required_years].sort_values("시도")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.info(
    "CSV 파일 3개를 앱 코드에 포함했습니다. 따라서 앱을 실행할 때 별도로 CSV를 업로드할 필요가 없습니다. "
    "학습 자료는 2021→2022, 2022→2023, 2023→2024의 변화량으로 구성됩니다."
)

# -------------------------------------------------------------------
# 3. 예측 변수 자동 설정
# -------------------------------------------------------------------
st.subheader("2. 예측 변수")
st.write("체크박스 없이 데이터에 포함된 모든 예측 변수를 자동으로 사용합니다.")

# 모든 독립변수를 자동으로 사용합니다.
selected_features = [
    "기준연도 사고건수",
    "기준연도",
    "직전 연도 증감",
    "직전 연도 증감률",
    "시도"
]

numeric_features = [f for f in selected_features if f != "시도"]
categorical_features = ["시도"]

# -------------------------------------------------------------------
# 4. 모델 생성 및 도시 단위 평가
# -------------------------------------------------------------------
X = model_df[selected_features].copy()
y = model_df["실제 전년 대비 증감"].astype(float)
groups = model_df["시도"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

transformers = []
if numeric_features:
    transformers.append(("num", numeric_transformer, numeric_features))
if categorical_features:
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers.append(("cat", categorical_transformer, categorical_features))

preprocessor = ColumnTransformer(transformers=transformers)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

st.subheader("3. 학습 및 성능 평가")
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
train_cities = sorted(model_df.iloc[train_idx]["시도"].unique().tolist())
test_cities = sorted(model_df.iloc[test_idx]["시도"].unique().tolist())

model.fit(X_train, y_train)
test_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, test_pred)
rmse = np.sqrt(mean_squared_error(y_test, test_pred))
r2 = r2_score(y_test, test_pred) if len(y_test) > 1 else np.nan

m1, m2, m3 = st.columns(3)
m1.metric("R² 결정계수", f"{r2:.3f}" if np.isfinite(r2) else "계산 불가")
m2.metric("MAE 평균 절대 오차", f"{mae:,.1f}건")
m3.metric("RMSE 평균 제곱근 오차", f"{rmse:,.1f}건")

st.write(f"**학습에 사용한 시도 ({len(train_cities)}개):** " + ", ".join(train_cities))
st.write(f"**평가에 사용한 시도 ({len(test_cities)}개):** " + ", ".join(test_cities))

st.info(
    "시도도 예측 변수에 포함되어 있습니다. 다만 평가 단계에서는 학습에 사용하지 않은 시도가 테스트에 포함되므로, "
    "처음 보는 시도 이름은 One-Hot 인코딩에서 별도의 학습 효과를 제공하지 않을 수 있습니다."
)

result_test = model_df.iloc[test_idx][["시도", "기준연도", "예측 대상 연도"]].copy()
result_test["실제 증감"] = y_test.to_numpy()
result_test["예측 증감"] = test_pred
result_test["오차"] = result_test["예측 증감"] - result_test["실제 증감"]
result_test["절대 오차"] = result_test["오차"].abs()
result_test = result_test.sort_values(["시도", "기준연도"])

with st.expander("학습·평가 데이터 구분 자세히 보기"):
    st.write("학습 자료와 평가 자료는 시도 단위로 나눴습니다. 같은 시도의 다른 연도가 학습 자료와 평가 자료에 동시에 들어가지 않도록 하여, 새로운 시도에 대한 예측 성능을 확인합니다.")
    st.dataframe(result_test.round(2), use_container_width=True, hide_index=True)

fig_eval = px.bar(
    result_test,
    x="시도",
    y=["실제 증감", "예측 증감"],
    barmode="group",
    color_discrete_sequence=["#4C78A8", "#F58518"],
    title="평가 데이터: 실제 증감과 예측 증감 비교",
    hover_data=["기준연도", "예측 대상 연도", "오차"]
)
fig_eval.add_hline(y=0, line_dash="dash", line_color="gray")
fig_eval.update_layout(xaxis_title="시도", yaxis_title="전년 대비 사고 건수 변화(건)")
st.plotly_chart(fig_eval, use_container_width=True)

# -------------------------------------------------------------------
# 5. 전체 데이터로 다시 학습하고 2025년 예측
# -------------------------------------------------------------------
st.subheader("4. 2025년 시도별 교통사고 건수 변화 예측")
st.caption("평가가 끝난 뒤, 2021~2024년의 모든 학습 자료로 모델을 다시 학습해 2025년 변화를 예측합니다.")

final_model = Pipeline(steps=[
    ("preprocessor", ColumnTransformer(transformers=transformers)),
    ("regressor", LinearRegression())
])
final_model.fit(X, y)

future_X = future_df[selected_features].copy()
future_pred = final_model.predict(future_X)

prediction = future_df[["시도", "기준연도 사고건수"]].copy()
prediction["예측 전년 대비 증감"] = np.rint(future_pred).astype(int)
prediction["예측 2025년 사고건수"] = (
    prediction["기준연도 사고건수"] + prediction["예측 전년 대비 증감"]
)
prediction["예측 방향"] = np.where(
    prediction["예측 전년 대비 증감"] > 0, "증가",
    np.where(prediction["예측 전년 대비 증감"] < 0, "감소", "변화 없음")
)
prediction = prediction.sort_values("시도").reset_index(drop=True)

st.dataframe(
    prediction.style.format({
        "기준연도 사고건수": "{:,.0f}",
        "예측 전년 대비 증감": "{:+,}건",
        "예측 2025년 사고건수": "{:,.0f}"
    }),
    use_container_width=True,
    hide_index=True
)

fig_pred = px.bar(
    prediction,
    x="시도",
    y="예측 전년 대비 증감",
    color="예측 전년 대비 증감",
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    title="시도별 2025년 전년 대비 교통사고 건수 예측"
)
fig_pred.add_hline(y=0, line_dash="dash", line_color="black")
fig_pred.update_layout(
    xaxis_title="시도",
    yaxis_title="예측한 사고 건수 변화(건)",
    coloraxis_showscale=False
)
st.plotly_chart(fig_pred, use_container_width=True)

st.download_button(
    "2025년 예측 결과 CSV 다운로드",
    data=prediction.to_csv(index=False, encoding="utf-8-sig"),
    file_name="시도별_2025년_교통사고_예측.csv",
    mime="text/csv"
)

st.subheader("5. 결과 해석 시 주의사항")
st.markdown(
    """
    - 예측값은 실제 2025년 통계가 아니라 2021~2024년의 변화 패턴을 바탕으로 계산한 추정치입니다.
    - 현재 데이터는 17개 시도와 3개 연도 변화만 포함하므로 학습 표본이 많지 않습니다.
    - R²가 낮거나 음수일 수 있으며, 이는 새로운 시도에 대한 예측이 실제값을 충분히 설명하지 못한다는 뜻입니다.
    - 교통사고 건수는 교통량, 인구, 도로 환경, 정책 등 다양한 요인의 영향을 받으므로 이 모델만으로 원인을 설명할 수는 없습니다.
    """
)
