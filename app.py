# app.py
import streamlit as st
import pandas as pd
import xgboost as xgb
import ta
import plotly.graph_objects as go
from src import config, live_data, scanner
import os

# Sayfa Ayarları
st.set_page_config(page_title="BIST Hisse Yön Tahmini", layout="wide")

st.title("📈 Borsa İstanbul Yapay Zeka Yön Tahmini")
st.markdown("""
Bu proje **XGBoost** algoritması kullanarak BIST 30 hisselerinin 
bir sonraki günkü kapanış yönünü (Artış/Düşüş) tahmin eder.
""")

# Yan Menü
st.sidebar.header("Hisse Seçimi")
selected_ticker = st.sidebar.selectbox(
    "Hisse Senedi Seçiniz",
    [t.replace(".IS", "") for t in config.TICKERS],
    format_func=lambda x: f"{x} (TRALT)" if x == "KOZAL" else (f"{x} (TRMET)" if x == "KOZAA" else x)
)
selected_ticker_full = selected_ticker + ".IS"

# Model Yükleme
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(config.MODEL_PATH)
    return model

# Canlı Veri Çekme ve İşleme Fonksiyonu
@st.cache_data(ttl=900, show_spinner=False)
def get_prediction_data(ticker):
    df_processed, df = live_data.fetch_live_frame(ticker)
    # Sadece en son günü al (Yarın için tahmin yapacağız)
    last_row = df_processed.iloc[[-1]]
    return last_row, df # df grafik çizimi için lazım


# Tüm BIST 30 Hisseleri için Önbellekli Tarama Fonksiyonu
@st.cache_data(ttl=900, show_spinner=False)
def get_cached_market_scan(tickers):
    model = load_model()
    return scanner.scan_market(tickers, model)


def render_single_ticker_tab():
    model = load_model()
    # Kullanıcı butona bastığında veya sayfa yüklendiğinde
    with st.spinner(f'{selected_ticker} verileri analiz ediliyor...'):
        input_data, full_df = get_prediction_data(selected_ticker_full)

        # Gerekli Featurelar
        features_list = ['rsi', 'macd', 'sma_10', 'sma_50', 'bb_width', 
                         'volatility', 'lag_1_ret', 'lag_2_ret', 'vol_change', 
                         'day_of_week', 'month']

        X_pred = input_data[features_list]

        # Tahmin
        prob = model.predict_proba(X_pred)[0][1] # Artış olasılığı
        prediction = 1 if prob > 0.5 else 0

        # GÖSTERGE PANELİ
        col1, col2, col3 = st.columns(3)

        # Fiyat ve Değişim Hesaplama
        if len(full_df) >= 2:
            current_price = full_df['close'].iloc[-1]
            prev_price = full_df['close'].iloc[-2]

            # Değişim Miktarı (TL) ve Oranı (%)
            change_amount = current_price - prev_price
            change_rate = (change_amount / prev_price) * 100
        else:
            current_price = full_df['close'].iloc[-1]
            change_amount = 0
            change_rate = 0

        with col1:
            # Delta color parametresini otomatikte bırakıyoruz, Streamlit +/- algılayıp renk verir
            st.metric(
                label=f"{selected_ticker} Son Fiyat",
                value=f"{current_price:.2f} TL",
                delta=f"{change_rate:.2f}%"  # Format: "-2.15%" veya "1.50%"
            )

        with col2:
            st.write("🤖 **Modelin Yarınki Tahmini:**") # Başlık ekledik ki karışmasın
            if prediction == 1:
                st.success(f"YÖN: **YUKARI** 🚀")
            else:
                st.error(f"YÖN: **DÜŞÜŞ / YATAY** 🔻")

        with col3:
            st.write("📊 **Güven Skoru:**")
            st.info(f"%{prob*100:.1f} Olasılıkla")

        # GRAFİK KISMI (Candlestick)
        st.subheader(f"{selected_ticker} - Son 3 Ay Fiyat Grafiği")
        fig = go.Figure(data=[go.Candlestick(x=full_df['Date'][-90:],
                        open=full_df['open'][-90:],
                        high=full_df['high'][-90:],
                        low=full_df['low'][-90:],
                        close=full_df['close'][-90:])])
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width='stretch')

        # Explainability (PDF Şartı: Neden bu karar?)
        st.subheader("Model Neden Bu Kararı Verdi?")
        st.write("Son günün teknik verileri:")
        st.dataframe(input_data[['rsi', 'macd', 'sma_10', 'sma_50', 'volatility']])

        if input_data['rsi'].values[0] < 30:
            st.markdown("- **RSI** aşırı satım bölgesinde (30 altı), bu genellikle tepki alımı geleceğine işaret edebilir.")
        elif input_data['rsi'].values[0] > 70:
            st.markdown("- **RSI** aşırı alım bölgesinde (70 üstü), düzeltme gelebilir.")


def render_market_scanner_tab():
    st.subheader("📊 BIST 30 Piyasa Fırsat Radarı")
    st.markdown("""
    Bu modül BIST 30 endeksindeki tüm hisseleri yapay zeka modelinden geçirerek 
    yarın için en yüksek yükseliş potansiyeline ve düşüş riskine sahip hisseleri sıralar.
    """)

    c_btn1, c_btn2 = st.columns([3, 7])
    with c_btn1:
        start_scan = st.button("🚀 BIST 30 Taramasını Başlat", key="btn_start_scan")
    with c_btn2:
        if st.session_state.get("market_scan_done", False):
            if st.button("🔄 Taramayı Yenile", key="btn_refresh_scan"):
                get_cached_market_scan.clear()
                st.session_state["market_scan_done"] = True
                st.rerun()

    if start_scan or st.session_state.get("market_scan_done", False):
        st.session_state["market_scan_done"] = True
        with st.spinner("BIST 30 hisseleri taranıyor ve analiz ediliyor..."):
            scan_df = get_cached_market_scan(tuple(config.TICKERS))

        if scan_df is not None and not scan_df.empty:
            top_bull = scan_df.head(5)
            top_bear = scan_df.tail(5).iloc[::-1]  # En düşükten yukarıya sırala

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 🟢 En Yüksek Yükseliş Potansiyeli (Top 5)")
                st.dataframe(
                    top_bull[["Hisse", "Son Fiyat (TL)", "Günlük Değişim (%)", "Yükseliş Olasılığı (%)"]],
                    hide_index=True
                )
            with c2:
                st.markdown("### 🔴 Düşüş Riski En Yüksek (Top 5)")
                st.dataframe(
                    top_bear[["Hisse", "Son Fiyat (TL)", "Günlük Değişim (%)", "Yükseliş Olasılığı (%)"]],
                    hide_index=True
                )

            st.markdown("### 📋 Tüm BIST 30 Liderlik Tablosu")
            st.dataframe(
                scan_df[["Hisse", "Son Fiyat (TL)", "Günlük Değişim (%)", "Yükseliş Olasılığı (%)", "Tahmin"]],
                hide_index=True
            )
        else:
            st.warning("Piyasa verileri taranırken veri alınamadı.")


def render_safely(render):
    """Bir sekmedeki hata yalnızca o sekmede gösterilir; diğer sekmeler çalışmaya devam eder."""
    try:
        render()
    except ValueError as e:
        st.warning(f"⚠️ {e}")
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")


# Ana Akış
if not os.path.exists(config.MODEL_PATH):
    st.error("Model dosyası bulunamadı! Lütfen önce `src/model_train.py` çalıştırın.")
else:
    tab1, tab2 = st.tabs(["🎯 Tek Hisse Analizi", "📊 BIST 30 Fırsat Radarı"])

    with tab1:
        render_safely(render_single_ticker_tab)

    with tab2:
        render_safely(render_market_scanner_tab)
