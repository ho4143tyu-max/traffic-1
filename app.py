import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="시도별 교통사고 건수 예측",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 시도별 교통사고 건수 예측")
st.caption("2016~2024년 한국도로교통공단 자료를 이용해 2027년 이후 시도별 연간 교통사고 건수를 예측합니다.")

# 업로드한 CSV 8개에서 연도별 사고 건수를 통합한 자료입니다.
# 모든 원본 파일의 기준월은 12월입니다.
ANNUAL_DATA = {'강원': {2016: 8841,
        2017: 8316,
        2018: 7498,
        2019: 7554,
        2020: 7039,
        2021: 6605,
        2022: 6425,
        2023: 6079,
        2024: 5930},
 '경기': {2016: 51013,
        2017: 50627,
        2018: 53448,
        2019: 55463,
        2020: 52391,
        2021: 53332,
        2022: 52968,
        2023: 53164,
        2024: 52175},
 '경남': {2016: 12522,
        2017: 11742,
        2018: 11493,
        2019: 12830,
        2020: 11739,
        2021: 11059,
        2022: 10611,
        2023: 11122,
        2024: 10703},
 '경북': {2016: 14629,
        2017: 13896,
        2018: 13966,
        2019: 14648,
        2020: 12673,
        2021: 12100,
        2022: 11037,
        2023: 10777,
        2024: 10403},
 '광주': {2016: 7354,
        2017: 7499,
        2018: 7459,
        2019: 8169,
        2020: 7718,
        2021: 7543,
        2022: 7122,
        2023: 6972,
        2024: 6908},
 '대구': {2016: 13098,
        2017: 12970,
        2018: 13199,
        2019: 14536,
        2020: 12940,
        2021: 12133,
        2022: 11102,
        2023: 10880,
        2024: 10222},
 '대전': {2016: 7535,
        2017: 7767,
        2018: 7554,
        2019: 8337,
        2020: 7215,
        2021: 7027,
        2022: 6841,
        2023: 7398,
        2024: 7490},
 '부산': {2016: 12192,
        2017: 11753,
        2018: 11937,
        2019: 12992,
        2020: 11913,
        2021: 11339,
        2022: 10882,
        2023: 11003,
        2024: 11255},
 '서울': {2016: 40039,
        2017: 38625,
        2018: 38795,
        2019: 39258,
        2020: 35227,
        2021: 33913,
        2022: 33698,
        2023: 33811,
        2024: 33465},
 '세종': {2016: 521, 2017: 746, 2018: 795, 2019: 922, 2020: 813, 2021: 788, 2022: 932, 2023: 1166, 2024: 1231},
 '울산': {2016: 4496,
        2017: 4265,
        2018: 3992,
        2019: 4347,
        2020: 3834,
        2021: 3801,
        2022: 3660,
        2023: 3664,
        2024: 3576},
 '인천': {2016: 8535,
        2017: 7719,
        2018: 7632,
        2019: 8698,
        2020: 8505,
        2021: 7912,
        2022: 7695,
        2023: 7648,
        2024: 8089},
 '전남': {2016: 9504,
        2017: 9770,
        2018: 9787,
        2019: 10877,
        2020: 9816,
        2021: 8685,
        2022: 8335,
        2023: 8287,
        2024: 8213},
 '전북': {2016: 7850,
        2017: 7748,
        2018: 6929,
        2019: 7615,
        2020: 6451,
        2021: 6112,
        2022: 5681,
        2023: 5830,
        2024: 5841},
 '제주': {2016: 4434,
        2017: 4378,
        2018: 4239,
        2019: 4412,
        2020: 4030,
        2021: 4373,
        2022: 4039,
        2023: 3950,
        2024: 3963},
 '충남': {2016: 8962,
        2017: 9241,
        2018: 8807,
        2019: 9404,
        2020: 8952,
        2021: 8423,
        2022: 8164,
        2023: 8606,
        2024: 8896},
 '충북': {2016: 9392,
        2017: 9273,
        2018: 9618,
        2019: 9538,
        2020: 8398,
        2021: 7985,
        2022: 7644,
        2023: 7939,
        2024: 7989}}

YEARS = list(range(2016, 2025))
REGIONS = sorted(ANNUAL_DATA.keys())
REFERENCE_MONTH = "12월"

# -------------------------------------------------------------------
# 1. 데이터프레임 구성
# -------------------------------------------------------------------
records = []
for region in REGIONS:
    for year in YEARS:
        if year in ANNUAL_DATA[region]:
            records.append({
                "연도": year,
                "기준월": REFERENCE_MONTH,
                "시도": region,
                "교통사고 건수": ANNUAL_DATA[region][year]
            })

data = pd.DataFrame(records).sort_values(["연도", "시도"]).reset_index(drop=True)

st.subheader("1. 데이터 개요")
c1, c2, c3 = st.columns(3)
c1.metric("분석 기간", f"{min(YEARS)}~{max(YEARS)}년")
c2.metric("분석 시도 수", f"{len(REGIONS)}개")
c3.metric("데이터 기준월", REFERENCE_MONTH)

st.info(
    "첨부된 자료는 월별 사고 건수가 아니라 연간 사고 건수 자료입니다. "
    "파일의 기준일이 모두 12월이므로 표에는 기준월을 '12월'로 표시합니다. "
    "이는 12월 한 달 동안의 사고 건수가 아니라 해당 연도의 연간 누계 건수를 뜻합니다."
)

# -------------------------------------------------------------------
# 2. 연도별 표
# -------------------------------------------------------------------
st.subheader("2. 연도별 시도 교통사고 건수")

selected_data_year = st.selectbox(
    "확인할 연도를 선택하세요.",
    YEARS,
    index=len(YEARS) - 1,
    format_func=lambda y: f"{y}년"
)

year_table = data[data["연도"] == selected_data_year][
    ["연도", "기준월", "시도", "교통사고 건수"]
].copy()

year_total = int(year_table["교통사고 건수"].sum())
st.metric(f"{selected_data_year}년 전체 시도 사고 건수 합계", f"{year_total:,}건")
st.dataframe(
    year_table.style.format({"교통사고 건수": "{:,.0f}건"}),
    use_container_width=True,
    hide_index=True
)

with st.expander("전체 연도의 시도별 사고 건수 한눈에 보기"):
    wide_table = data.pivot(index="시도", columns="연도", values="교통사고 건수").reset_index()
    wide_table.columns.name = None
    wide_table = wide_table.sort_values("시도")
    st.dataframe(wide_table, use_container_width=True, hide_index=True)

with st.expander("연도별 데이터 기준월 확인"):
    month_table = (
        data[["연도", "기준월"]]
        .drop_duplicates()
        .sort_values("연도")
        .reset_index(drop=True)
    )
    st.dataframe(month_table, use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# 3. 연도별 추이 시각화
# -------------------------------------------------------------------
st.subheader("3. 연도별 교통사고 건수 추이")
trend = data.groupby("연도", as_index=False)["교통사고 건수"].sum()
fig_trend = px.line(
    trend,
    x="연도",
    y="교통사고 건수",
    markers=True,
    title="전국 시도 사고 건수 합계의 연도별 추이"
)
fig_trend.update_layout(
    xaxis_title="연도",
    yaxis_title="시도별 사고 건수 합계(건)"
)
st.plotly_chart(fig_trend, use_container_width=True)

# -------------------------------------------------------------------
# 4. 미래 연도 예측
# -------------------------------------------------------------------
st.subheader("4. 2027년 이후 교통사고 건수 예측")
st.write(
    "각 시도별로 2016~2024년의 모든 연도별 사고 건수를 학습합니다. "
    "연도를 독립변수, 교통사고 건수를 종속변수로 사용하는 선형회귀 모델입니다."
)

prediction_year = st.number_input(
    "예측할 연도를 입력하세요.",
    min_value=2027,
    max_value=2100,
    value=2027,
    step=1
)

prediction_rows = []
for region in REGIONS:
    region_data = data[data["시도"] == region].sort_values("연도")
    X = region_data[["연도"]]
    y = region_data["교통사고 건수"]

    model = LinearRegression()
    model.fit(X, y)

    predicted_count = float(model.predict(pd.DataFrame({"연도": [int(prediction_year)]}))[0])
    predicted_count = max(0, int(round(predicted_count)))

    prediction_rows.append({
        "시도": region,
        "예측 연도": int(prediction_year),
        "기준월": REFERENCE_MONTH,
        "예측 교통사고 건수": predicted_count
    })

prediction = pd.DataFrame(prediction_rows).sort_values("시도").reset_index(drop=True)
total_prediction = int(prediction["예측 교통사고 건수"].sum())

p1, p2 = st.columns(2)
p1.metric(f"{int(prediction_year)}년 예측 사고 건수 합계", f"{total_prediction:,}건")
p2.metric("예측 대상 시도", f"{len(prediction)}개")

st.dataframe(
    prediction.style.format({"예측 교통사고 건수": "{:,.0f}건"}),
    use_container_width=True,
    hide_index=True
)

fig_prediction = px.bar(
    prediction,
    x="시도",
    y="예측 교통사고 건수",
    title=f"{int(prediction_year)}년 시도별 교통사고 건수 예측",
    color="예측 교통사고 건수",
    color_continuous_scale="Blues"
)
fig_prediction.update_layout(
    xaxis_title="시도",
    yaxis_title="예측 교통사고 건수(건)",
    coloraxis_showscale=False
)
st.plotly_chart(fig_prediction, use_container_width=True)

st.download_button(
    label="예측 결과 CSV 다운로드",
    data=prediction.to_csv(index=False, encoding="utf-8-sig"),
    file_name=f"{int(prediction_year)}년_시도별_교통사고_예측.csv",
    mime="text/csv"
)

# -------------------------------------------------------------------
# 5. 해석 및 주의사항
# -------------------------------------------------------------------
st.subheader("5. 결과 해석 시 주의사항")
st.markdown(
    """
    - 원본 자료는 2016~2024년의 연간 교통사고 건수입니다.
    - 기준월 '12월'은 자료의 기준일이 12월이라는 뜻이며, 12월 한 달의 사고 건수를 의미하지 않습니다.
    - 각 시도별 선형회귀 모델은 해당 시도의 2016~2024년 자료 전체를 학습합니다.
    - 2027년 이후 예측은 과거 추세가 계속된다는 가정에 따른 추정값입니다.
    - 예측 기간이 멀어질수록 실제 결과와 차이가 커질 수 있습니다.
    - 교통량, 인구, 도로 환경, 교통 정책 등 다른 요인은 모델에 포함되지 않았습니다.
    """
)
