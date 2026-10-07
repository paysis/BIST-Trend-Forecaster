# app.py
import streamlit as st
import pandas as pd
import xgboost as xgb
import yfinance as yf
import ta
import plotly.graph_objects as go
from src import config, explain, features, scanner, signals
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

st.sidebar.header("Sinyal Ayarları")
min_confidence = st.sidebar.slider(
    "Minimum Güven Eşiği",
    min_value=0.51,
    max_value=0.70,
    value=config.PROB_THRESHOLD_HIGH,
    step=0.01,
    help="Olasılık bu eşiğin üzerinde (ya da 1 - eşik altında) değilse sinyal nötr gösterilir.",
)
prob_low, prob_high = signals.symmetric_band(min_confidence)

# Model Yükleme
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(config.MODEL_PATH)
    return model

# Canlı Veri Çekme ve İşleme Fonksiyonu
@st.cache_data(ttl=900, show_spinner=False)
def get_prediction_data(ticker):
    # Yahoo Finance sembolü değişen hisseleri eşle (Örn: KOZAL -> TRALT)
    yahoo_ticker = getattr(config, "TICKER_YAHOO_MAP", {}).get(ticker, ticker)
    
    # Modelin indikatörleri hesaplayabilmesi için son 6 ayın verisine ihtiyacı var
    df = yf.download(yahoo_ticker, period="6mo", progress=False)
    
    if df is None or df.empty or len(df) == 0:
        raise ValueError(
            f"'{ticker}' (Yahoo: '{yahoo_ticker}') için piyasa verisi alınamadı. "
            "Sembol değişmiş veya Yahoo Finance servisi yanıt vermiyor olabilir."
        )
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # DÜZELTME BURADA: 'Ticker' yerine küçük harfle 'ticker' yaptık
    df['ticker'] = ticker.replace(".IS", "")
    
    df.reset_index(inplace=True)
    
    # Sütun isimlerini düzenle (features.py 'Date' ve küçük harfli sütunlar bekliyor)
    new_columns = {}
    for col in df.columns:
        if col.lower() in ('date', 'index'):
            new_columns[col] = 'Date' # Date büyük kalsın
        elif col.lower() == 'ticker':
            new_columns[col] = 'ticker' # ticker küçük kalsın
        else:
            new_columns[col] = col.lower() # open, close, high, low vs. küçük olsun
            
    df.rename(columns=new_columns, inplace=True)
    
    if 'Date' not in df.columns:
        raise ValueError(f"'{ticker}' için çekilen veride 'Date' sütunu bulunamadı.")
    
    # Feature Engineering Scriptini Kullan
    # drop_incomplete_target=False: canlı tahminde bugünün hedefi (yarının kapanışı)
    # henüz bilinmez; bu normalde eğitimde düşürülen son günü burada tutar.
    df_processed = features.add_features(df, drop_incomplete_target=False)
    
    if df_processed.empty:
        raise ValueError(f"'{ticker}' verisi teknik indikatörler hesaplandıktan sonra yetersiz kaldı.")

    # Sadece en son günü al (Yarın için tahmin yapacağız)
    last_row = df_processed.iloc[[-1]]
    return last_row, df # df grafik çizimi için lazım


# Tüm BIST 30 Hisseleri için Önbellekli Tarama Fonksiyonu
@st.cache_data(ttl=900, show_spinner=False)
def get_cached_market_scan(tickers):
    model = load_model()
    return scanner.scan_market(tickers, model)


# Ana Akış
try:
    if not os.path.exists(config.MODEL_PATH):
        st.error("Model dosyası bulunamadı! Lütfen önce `src/model_train.py` çalıştırın.")
    else:
        model = load_model()
        tab1, tab2 = st.tabs(["🎯 Tek Hisse Analizi", "📊 BIST 30 Fırsat Radarı"])

        with tab1:
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
                signal = signals.classify_signal(prob, prob_low, prob_high)
                # Sinyale göre renkli kutu: yeşil (yükseliş), sarı (nötr), kırmızı (düşüş)
                signal_box, signal_label = {
                    signals.UP: (st.success, "**YÜKSELİŞ BEKLENTİSİ** 🚀"),
                    signals.NEUTRAL: (st.warning, "**NÖTR / BELİRSİZ PİYASA** ⏸️"),
                    signals.DOWN: (st.error, "**DÜŞÜŞ / ZAYIF TREND** 🔻"),
                }[signal]
                
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
                    signal_box(signal_label)
                
                with col3:
                    st.write("📊 **Güven Skoru:**")
                    signal_box(f"%{prob*100:.1f} Olasılıkla Yükseliş")

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
                # Modelin gerçek TreeSHAP katkıları: her özniteliğin bu tahmindeki payı
                st.subheader("Model Neden Bu Kararı Verdi?")
                explanation, base_prob = explain.explain_prediction(model, X_pred)
                st.markdown(explain.summarize_drivers(explanation, selected_ticker))

                impacts = explanation['impact'] * 100
                levels = base_prob * 100 + impacts.cumsum()
                fig_explain = go.Figure(go.Waterfall(
                    measure=["absolute"] + ["relative"] * len(explanation) + ["total"],
                    x=["Ortalama"] + [explain.FEATURE_LABELS.get(f, f) for f in explanation['feature']] + ["Tahmin"],
                    y=[base_prob * 100] + impacts.tolist() + [0],
                    text=[f"%{base_prob * 100:.1f}"] + [f"{i:+.1f}" for i in impacts] + [f"%{prob * 100:.1f}"],
                    increasing={"marker": {"color": "#2ca02c"}},
                    decreasing={"marker": {"color": "#d62728"}},
                ))
                # Başlangıç çubuğu %0'dan başlarsa birkaç puanlık katkılar okunmaz; ekseni yakınlaştır
                low, high = min(levels.min(), base_prob * 100), max(levels.max(), base_prob * 100)
                fig_explain.update_layout(
                    yaxis_title="Yükseliş Olasılığı (%)",
                    yaxis_range=[low - 2, high + 2],
                    showlegend=False,
                )
                st.plotly_chart(fig_explain, width='stretch')
                st.caption(
                    "Ortalama: modelin hiçbir göstergeyi bilmeden verdiği olasılık. "
                    "Her çubuk, ilgili göstergenin bugünkü değerinin olasılığı kaç puan "
                    "artırdığını (yeşil) ya da azalttığını (kırmızı) gösterir."
                )

                st.write("Son günün teknik verileri:")
                st.dataframe(input_data[['rsi', 'macd', 'sma_10', 'sma_50', 'volatility']])

        with tab2:
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

except ValueError as e:
    st.warning(f"⚠️ {e}")
except Exception as e:
    st.error(f"Bir hata oluştu: {e}")