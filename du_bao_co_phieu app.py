# du_bao_co_phieu
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# =========================
# 1. CẤU HÌNH GIAO DIỆN
# =========================
st.set_page_config(
    page_title="AI Stock Forecast",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #08111f, #101d30);
    color: #f4f7fb;
}
h1, h2, h3 {
    color: #eaf4ff;
}
[data-testid="stMetric"] {
    background: #14243a;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263b55;
}
div.stButton > button[kind="primary"] {
    background: #1687ff;
    border-radius: 10px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 AI STOCK FORECAST")
st.caption("Phân tích dữ liệu và dự báo giá cổ phiếu bằng Machine Learning")

# =========================
# 2. NHẬP THÔNG TIN
# =========================
with st.sidebar:
    st.header("⚙️ Cấu hình dự báo")

    ticker = st.text_input(
        "Mã cổ phiếu",
        value="FPT.VN",
        help="Ví dụ Việt Nam: FPT.VN, VNM.VN, HPG.VN. Mỹ: AAPL, MSFT."
    ).strip().upper()

    history = st.selectbox(
        "Dữ liệu lịch sử",
        ["1 năm", "2 năm", "5 năm"],
        index=2
    )

    forecast_days = st.slider(
        "Số phiên muốn dự báo",
        min_value=5,
        max_value=30,
        value=10,
        step=5
    )

    run = st.button(
        "🚀 Phân tích và dự báo",
        type="primary",
        use_container_width=True
    )

st.info(
    "Dữ liệu lấy từ Yahoo Finance qua yfinance. "
    "Một số mã cổ phiếu có thể không được hỗ trợ."
)

# =========================
# 3. TẢI DỮ LIỆU
# =========================
@st.cache_data(ttl=1800)
def get_stock_data(symbol, period):
    data = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    # Xử lý tên cột trong các phiên bản yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        return pd.DataFrame()

    result = data[["Close"]].copy()
    result["Close"] = pd.to_numeric(
        result["Close"], errors="coerce"
    )
    result = result.dropna()
    result.index = pd.to_datetime(result.index)

    return result[~result.index.duplicated(keep="last")]


# =========================
# 4. TẠO CHỈ BÁO KỸ THUẬT
# =========================
FEATURES = [
    "ret1", "ret2", "ret5", "ret10",
    "ma5", "ma20", "ma50",
    "vol10", "vol20", "rsi14"
]


def make_features(close):
    close = close.astype(float)

    f = pd.DataFrame(index=close.index)
f["ret1"] = close.pct_change(1)
    f["ret2"] = close.pct_change(2)
    f["ret5"] = close.pct_change(5)
    f["ret10"] = close.pct_change(10)

    f["ma5"] = close / close.rolling(5).mean() - 1
    f["ma20"] = close / close.rolling(20).mean() - 1
    f["ma50"] = close / close.rolling(50).mean() - 1

    f["vol10"] = f["ret1"].rolling(10).std()
    f["vol20"] = f["ret1"].rolling(20).std()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    f["rsi14"] = 100 - 100 / (1 + rs)

    # Trường hợp giá tăng liên tục
    f.loc[avg_loss == 0, "rsi14"] = 100

    return f.replace([np.inf, -np.inf], np.nan)


# =========================
# 5. HUẤN LUYỆN VÀ DỰ BÁO
# =========================
def forecast_stock(data, days):
    close = data["Close"].astype(float)
    features = make_features(close)

    # Dự đoán tỷ suất sinh lợi của phiên kế tiếp
    target = close.shift(-1) / close - 1

    dataset = features.copy()
    dataset["target"] = target
    dataset = dataset.dropna()

    if len(dataset) < 100:
        raise ValueError(
            "Không đủ dữ liệu để huấn luyện. "
            "Hãy chọn mã khác hoặc khoảng lịch sử dài hơn."
        )

    # Chia tập theo thời gian, không xáo trộn dữ liệu
    split = int(len(dataset) * 0.8)

    X_train = dataset[FEATURES].iloc[:split]
    y_train = dataset["target"].iloc[:split]

    X_test = dataset[FEATURES].iloc[split:]
    y_test = dataset["target"].iloc[split:]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Kiểm tra trên dữ liệu chưa dùng để huấn luyện
    test_base_prices = close.loc[X_test.index]
    actual_prices = test_base_prices * (1 + y_test)
    predicted_prices = test_base_prices * (
        1 + model.predict(X_test)
    )

    mae = mean_absolute_error(
        actual_prices, predicted_prices
    )

    # Huấn luyện lại bằng toàn bộ dữ liệu có nhãn
    model.fit(
        dataset[FEATURES],
        dataset["target"]
    )

    # Dự báo lần lượt từng phiên tiếp theo
    future_prices = []
    last_date = close.index[-1]
    last_price = float(close.iloc[-1])

    simulated_close = close.copy()

    for _ in range(days):
        current_features = make_features(
            simulated_close
        ).dropna()

        if current_features.empty:
            raise ValueError(
                "Không thể tạo chỉ báo kỹ thuật cho dữ liệu."
            )

        x_now = current_features[FEATURES].iloc[[-1]]
predicted_return = float(model.predict(x_now)[0])

        # Giới hạn bất thường để tránh dự báo mất ổn định
        predicted_return = float(
            np.clip(predicted_return, -0.15, 0.15)
        )

        last_price = last_price * (1 + predicted_return)

        next_date = last_date + pd.offsets.BDay(1)
        last_date = next_date

        simulated_close.loc[next_date] = last_price

        future_prices.append({
            "Ngày dự báo": next_date.strftime("%Y-%m-%d"),
            "Giá dự báo": last_price
        })

    forecast = pd.DataFrame(future_prices)

    return forecast, mae, predicted_prices, actual_prices


# =========================
# 6. HIỂN THỊ KẾT QUẢ
# =========================
if run:
    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    period_map = {
        "1 năm": "1y",
        "2 năm": "2y",
        "5 năm": "5y"
    }

    try:
        with st.spinner(
            "Đang tải dữ liệu và huấn luyện mô hình AI..."
        ):
            data = get_stock_data(
                ticker, period_map[history]
            )

            if data.empty:
                st.error(
                    "Không tìm thấy dữ liệu cho mã này. "
                    "Hãy kiểm tra mã cổ phiếu."
                )
                st.stop()

            forecast, mae, pred_test, actual_test = (
                forecast_stock(data, forecast_days)
            )

        st.success(f"Đã hoàn tất phân tích mã {ticker}")

        last_close = float(data["Close"].iloc[-1])
        predicted_close = float(forecast["Giá dự báo"].iloc[-1])

        change_pct = (
            (predicted_close / last_close) - 1
        ) * 100

        # Thẻ thông tin
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Giá đóng cửa gần nhất",
            f"{last_close:,.2f}"
        )

        c2.metric(
            f"Dự báo sau {forecast_days} phiên",
            f"{predicted_close:,.2f}",
            f"{change_pct:+.2f}%"
        )

        c3.metric(
            "MAE kiểm tra",
            f"{mae:,.2f}"
        )

        c4.metric(
            "Số phiên dự báo",
            forecast_days
        )

        st.caption(
            "Giá hiển thị theo đơn vị của nguồn dữ liệu. "
            "MAE càng thấp thường càng tốt khi so sánh "
            "trên cùng mã và cùng cách kiểm tra."
        )

        # Biểu đồ lịch sử và dự báo
        st.subheader("📊 Biểu đồ giá và dự báo")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["Close"],
            name="Giá lịch sử",
            mode="lines",
            line=dict(color="#43B7FF", width=2)
        ))

        forecast_dates = pd.to_datetime(
            forecast["Ngày dự báo"]
        )
fig.add_trace(go.Scatter(
            x=[data.index[-1]] + list(forecast_dates),
            y=[last_close] + list(forecast["Giá dự báo"]),
            name="Dự báo AI",
            mode="lines+markers",
            line=dict(color="#35D39A", width=2, dash="dash")
        ))

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Ngày",
            yaxis_title="Giá đóng cửa",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Bảng kết quả
        st.subheader("📅 Bảng giá dự báo")

        display_forecast = forecast.copy()
        display_forecast["Giá dự báo"] = (
            display_forecast["Giá dự báo"].round(2)
        )

        st.dataframe(
            display_forecast,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Tải bảng dự báo CSV",
            data=forecast.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
            mime="text/csv"
        )

        # Đánh giá mô hình
        st.subheader("🧠 Đánh giá mô hình")

        st.write(
            f"- Thuật toán: Random Forest Regressor.\n"
            f"- Dữ liệu lịch sử: {len(data)} phiên.\n"
            f"- Số mẫu kiểm tra: {len(actual_test)}.\n"
            f"- Sai số MAE: {mae:,.
