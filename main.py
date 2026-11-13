
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="시도별 전년대비 교통사고 건수 예측",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 시도별 전년대비 교통사고 건수 예측")
st.caption(
    "한국도로교통공단 교통사고 데이터를 활용한 다중 선형 회귀 분석"
)

st.markdown("""
2023년부터 2025년까지의 시도별 교통사고 데이터를 이용하여
전년 대비 교통사고 건수의 증가 또는 감소를 예측합니다.
""")


# ==========================================
# 2. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="cp949")



uploaded_files = st.file_uploader(
    "2023년~2025년 교통사고 CSV 파일을 선택하세요.",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:
    data_list = []

    for file in uploaded_files:
        temp = load_data(file)
        data_list.append(temp)

    raw = pd.concat(data_list, ignore_index=True)
    data_source = "업로드한 CSV 파일 3개"

else:
    try:
        raw = load_data("traffic.csv")
        data_source = "traffic.csv"
    except FileNotFoundError:
        st.warning("CSV 파일을 업로드해 주세요.")
        st.stop()



# ==========================================
# 3. 데이터 컬럼 자동 확인
# ==========================================

raw.columns = [
    str(c).strip().replace(" ", "")
    for c in raw.columns
]

def find_column(columns, keywords):
    for col in columns:
        if any(word in col for word in keywords):
            return col
    return None


region_col = find_column(
    raw.columns,
    ["시도", "시도명", "지역명", "지역"]
)

year_col = find_column(
    raw.columns,
    ["발생년도", "발생연도", "연도", "년도"]
)

count_col = find_column(
    raw.columns,
    ["사고건수", "교통사고건수"]
)

if region_col is None:
    st.error(
        "지역을 나타내는 컬럼을 찾을 수 없습니다. "
        "시도 또는 시도명 컬럼이 필요합니다."
    )
    st.stop()


# ==========================================
# 4. 데이터 형태 통일
# ==========================================

try:

    # 연도별 컬럼이 가로로 있는 형태
    year_columns = {}

    for col in raw.columns:
        for year in [2023, 2024, 2025]:
            if str(year) in col:
                year_columns[year] = col

    if all(year in year_columns for year in [2023, 2024, 2025]):

        annual = raw[
            [region_col] +
            [year_columns[y] for y in [2023, 2024, 2025]]
        ].copy()

        annual.columns = [
            "시도", "2023년", "2024년", "2025년"
        ]

        annual = annual.melt(
            id_vars="시도",
            var_name="연도",
            value_name="사고건수"
        )

        annual["연도"] = (
            annual["연도"].str.extract(r"(2023|2024|2025)")
            .astype(int)
        )

    # 연도와 사고건수가 세로로 있는 형태
    elif year_col is not None and count_col is not None:

        annual = raw[
            [region_col, year_col, count_col]
        ].copy()

        annual.columns = ["시도", "연도", "사고건수"]

        annual["연도"] = pd.to_numeric(
            annual["연도"].astype(str).str.extract(
                r"(2023|2024|2025)"
            )[0],
            errors="coerce"
        )

    else:
        st.error(
            "2023년, 2024년, 2025년 자료를 찾지 못했습니다.\n\n"
            "CSV에는 시도별 3개 연도의 사고 건수가 필요합니다."
        )
        st.stop()

except Exception as e:
    st.error(f"데이터 구조를 정리하는 중 오류가 발생했습니다: {e}")
    st.stop()


# ==========================================
# 5. 데이터 전처리
# ==========================================

annual["시도"] = (
    annual["시도"]
    .astype(str)
    .str.strip()
)

annual["사고건수"] = pd.to_numeric(
    annual["사고건수"]
    .astype(str)
    .str.replace(",", "", regex=False),
    errors="coerce"
)

annual["연도"] = pd.to_numeric(
    annual["연도"],
    errors="coerce"
)

annual = annual.dropna(
    subset=["시도", "연도", "사고건수"]
)

annual["연도"] = annual["연도"].astype(int)

annual = annual[
    annual["연도"].isin([2023, 2024, 2025])
]

# 전국 합계 행은 지역별 학습에서 제외
annual = annual[
    annual["시도"] != "전국"
]

# 동일 지역·연도 중복 데이터 확인 후 합산
annual = (
    annual.groupby(["시도", "연도"], as_index=False)
    ["사고건수"].sum()
)

# 3개 연도가 모두 존재하는 지역만 사용
pivot = annual.pivot(
    index="시도",
    columns="연도",
    values="사고건수"
)

required_years = [2023, 2024, 2025]

if not all(y in pivot.columns for y in required_years):
    st.error("2023~2025년 자료가 모두 필요합니다.")
    st.stop()

pivot = pivot.dropna(
    subset=required_years
)

if len(pivot) < 5:
    st.error(
        "분석 가능한 지역이 너무 적습니다. "
        "여러 시도의 3개 연도 자료를 확인하세요."
    )
    st.stop()

pivot = pivot.sort_index()

# 숫자형 연도 컬럼 유지
pivot.columns.name = None


# ==========================================
# 6. 전년 대비 증감 데이터 생성
# ==========================================

records = []

for region, row in pivot.iterrows():

    for year in [2024, 2025]:

        previous_year = year - 1

        previous_count = row[previous_year]
        current_count = row[year]

        # 예측 대상: 전년 대비 사고 건수 증감
        target = current_count - previous_count

        # 직전 연도의 증감
        if previous_year - 1 in pivot.columns:
            before_count = row[previous_year - 1]
            previous_change = previous_count - before_count

            previous_rate = (
                previous_change / before_count * 100
                if before_count != 0 else np.nan
            )
        else:
            previous_change = np.nan
            previous_rate = np.nan

        records.append({
            "시도": region,
            "기준연도": previous_year,
            "예측연도": year,
            "기준연도 사고건수": previous_count,
            "직전 연도 증감": previous_change,
            "직전 연도 증감률": previous_rate,
            "실제 증감 건수": target
        })

model_data = pd.DataFrame(records)

# 가나다순 정렬
model_data = model_data.sort_values(
    ["시도", "기준연도"]
).reset_index(drop=True)


# ==========================================
# 7. 데이터 정보
# ==========================================

st.subheader("📊 데이터 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("사용 지역 수", f"{pivot.shape[0]}개")

with col2:
    st.metric("사용 연도", "2023~2025년")

with col3:
    st.metric("연도별 학습 자료", f"{len(model_data)}개")

with col4:
    st.metric("데이터 출처", "한국도로교통공단")

st.caption(f"데이터 파일: {data_source}")

with st.expander("연도별 원본 데이터 확인"):
    display_pivot = pivot.reset_index()
    display_pivot.columns = [
        "시도", "2023년", "2024년", "2025년"
    ]
    st.dataframe(
        display_pivot,
        use_container_width=True,
        hide_index=True
    )

with st.expander("전년 대비 증감 데이터 확인"):
    st.dataframe(
        model_data,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 8. 예측 변수 선택
# ==========================================

st.sidebar.header("⚙️ 모델 설정")

st.sidebar.subheader("사용할 독립변수")

feature_options = {
    "기준연도 사고건수": True,
    "기준연도": True,
    "직전 연도 증감": True,
    "직전 연도 증감률": True,
    "시도": False
}

selected_features = []

for feature, default in feature_options.items():

    if st.sidebar.checkbox(
        feature,
        value=default,
        key=f"feature_{feature}"
    ):
        selected_features.append(feature)

if len(selected_features) == 0:
    st.error("최소 한 개 이상의 독립변수를 선택하세요.")
    st.stop()


# ==========================================
# 9. 학습 및 테스트 데이터 분리
# ==========================================

st.sidebar.subheader("학습 설정")

test_size = st.sidebar.slider(
    "테스트 도시 비율",
    min_value=0.2,
    max_value=0.4,
    value=0.25,
    step=0.05
)

random_state = st.sidebar.number_input(
    "데이터 분리 기준값",
    min_value=0,
    max_value=9999,
    value=42
)

X = model_data[selected_features]

y = model_data["실제 증감 건수"]

groups = model_data["시도"]

# 같은 도시의 두 연도 자료가 학습과 테스트에
# 동시에 들어가지 않도록 도시 단위로 분리
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=test_size,
    random_state=int(random_state)
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_data = model_data.iloc[train_idx]
test_data = model_data.iloc[test_idx]

train_cities = sorted(
    train_data["시도"].unique().tolist()
)

test_cities = sorted(
    test_data["시도"].unique().tolist()
)


# ==========================================
# 10. 다중 회귀 모델 구성
# ==========================================

numeric_features = [
    col for col in selected_features
    if col != "시도"
]

categorical_features = [
    col for col in selected_features
    if col == "시도"
]

transformers = []

if numeric_features:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# ==========================================
# 11. 예측 결과 계산
# ==========================================

result = test_data[
    ["시도", "기준연도", "예측연도", "실제 증감 건수"]
].copy()

result["예측 증감 건수"] = predictions

# 예측값과 실제값의 차이
result["오차"] = (
    result["예측 증감 건수"] -
    result["실제 증감 건수"]
)

result["절대 오차"] = result["오차"].abs()

result["예측 증감 건수"] = (
    result["예측 증감 건수"].round(2)
)

result["오차"] = result["오차"].round(2)

result["절대 오차"] = result["절대 오차"].round(2)

result = result.sort_values(
    ["시도", "기준연도"]
).reset_index(drop=True)


# ==========================================
# 12. 모델 평가
# ==========================================

r2 = (
    r2_score(y_test, predictions)
    if len(y_test) >= 2 else np.nan
)

mae = mean_absolute_error(
    y_test, predictions
)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

st.divider()

st.subheader("📈 모델 예측 성능")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "결정계수 R²",
        f"{r2:.3f}" if np.isfinite(r2) else "계산 불가"
    )

with m2:
    st.metric(
        "평균 절대 오차 MAE",
        f"{mae:,.2f}건"
    )

with m3:
    st.metric(
        "평균 제곱근 오차 RMSE",
        f"{rmse:,.2f}건"
    )

st.caption("""
R²는 실제 증감량의 변동을 모델이 얼마나 설명하는지 나타냅니다.
1에 가까울수록 좋으며, 음수도 나올 수 있습니다.

MAE와 RMSE는 예측값과 실제값의 평균적인 차이를 나타냅니다.
값이 작을수록 예측 오차가 작습니다.
""")

if not np.isfinite(r2):
    st.warning("테스트 데이터가 2개 미만이어서 R²를 계산할 수 없습니다.")

# 학습에 사용한 도시 표시
st.subheader("🏙️ 학습 및 테스트 도시")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**학습에 사용한 도시**")
    st.write(", ".join(train_cities))
    st.caption(f"총 {len(train_cities)}개 도시")

with col2:
    st.markdown("**학습에 사용하지 않은 테스트 도시**")
    st.write(", ".join(test_cities))
    st.caption(f"총 {len(test_cities)}개 도시")


# ==========================================
# 13. 학습 조건 표시
# ==========================================

st.subheader("🔍 학습 조건")

st.markdown(f"""
- **모델:** 다중 선형 회귀
- **학습 기간:** 2023년 → 2024년, 2024년 → 2025년
- **예측 대상:** 전년 대비 교통사고 건수 증감(건)
- **사용 독립변수:** {", ".join(selected_features)}
- **학습 도시 수:** {len(train_cities)}개
- **테스트 도시 수:** {len(test_cities)}개
- **학습 데이터 수:** {len(X_train)}개
- **테스트 데이터 수:** {len(X_test)}개
- **데이터 분리 기준값:** {random_state}
""")

st.info(
    "테스트 도시의 자료는 모델 학습에 사용하지 않았으며, "
    "해당 도시들의 실제값과 예측값을 비교하여 성능을 평가합니다."
)


# ==========================================
# 14. 예측 결과 표
# ==========================================

st.divider()

st.subheader("📋 도시별 예측 결과")

st.markdown("""
- 실제 증감 건수: 실제 교통사고 건수의 전년 대비 변화
- 예측 증감 건수: 회귀 모델이 예측한 변화
- 오차: 예측값 - 실제값
- 절대 오차: 오차의 절댓값
""")

st.dataframe(
    result.style.format({
        "실제 증감 건수": "{:,.0f}",
        "예측 증감 건수": "{:,.2f}",
        "오차": "{:,.2f}",
        "절대 오차": "{:,.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

# 결과 CSV 다운로드
csv = result.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    "📥 예측 결과 CSV 다운로드",
    data=csv,
    file_name="traffic_prediction_result.csv",
    mime="text/csv"
)


# ==========================================
# 15. Plotly 예측 그래프
# ==========================================

st.divider()

st.subheader("📊 도시별 예측 결과 그래프")

period_options = {
    "2023년 → 2024년": (2023, 2024),
    "2024년 → 2025년": (2024, 2025)
}

selected_period = st.selectbox(
    "그래프에 표시할 예측 기간",
    list(period_options.keys())
)

base_year, target_year = period_options[selected_period]

chart_data = result[
    (result["기준연도"] == base_year) &
    (result["예측연도"] == target_year)
].copy()

chart_data = chart_data.sort_values("시도")

if len(chart_data) == 0:
    st.warning(
        "선택한 기간에 테스트 도시의 예측 결과가 없습니다."
    )
else:

    fig = go.Figure()

    # 예측값 막대그래프
    fig.add_trace(
        go.Bar(
            x=chart_data["시도"],
            y=chart_data["예측 증감 건수"],
            name="예측 증감 건수",
            marker_color="#4C9BE8",
            text=chart_data["예측 증감 건수"],
            texttemplate="%{text:,.1f}",
            textposition="outside"
        )
    )

    # 실제값 기준선
    fig.add_trace(
        go.Scatter(
            x=chart_data["시도"],
            y=chart_data["실제 증감 건수"],
            name="실제 증감 건수 (기준)",
            mode="lines+markers",
            line=dict(
                color="#E45756",
                width=3
            ),
            marker=dict(size=9)
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="증감 없음"
    )

    fig.update_layout(
        title=f"{base_year}년 대비 {target_year}년 교통사고 건수 예측",
        xaxis_title="도시",
        yaxis_title="전년 대비 교통사고 건수 증감 (건)",
        template="plotly_white",
        height=600,
        barmode="group",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=chart_data["시도"].tolist(),
        tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption("""
    파란색 막대는 모델의 예측값이며,
    빨간색 선은 실제 교통사고 건수의 증감입니다.
    두 값의 차이가 작을수록 예측이 실제에 가까운 것입니다.
    """)


# ==========================================
# 16. 오차 분석
# ==========================================

st.divider()

st.subheader("📉 예측 오차 분석")

error_col1, error_col2 = st.columns(2)

with error_col1:
    st.metric(
        "가장 큰 절대 오차",
        f"{result['절대 오차'].max():,.2f}건"
    )

with error_col2:
    st.metric(
        "평균 절대 오차",
        f"{result['절대 오차'].mean():,.2f}건"
    )

worst = result.loc[
    result["절대 오차"].idxmax()
]

st.write(
    f"가장 큰 오차가 발생한 지역은 **{worst['시도']}**이며, "
    f"예측과 실제의 차이는 **{worst['절대 오차']:,.2f}건**입니다."
)

st.caption(
    "이 모델은 3개 연도의 제한된 자료를 사용하므로 "
    "미래 교통사고 건수를 확정적으로 예측하는 도구가 아니라 "
    "데이터 분석과 회귀 모델 학습을 위한 실습 도구입니다."
)
