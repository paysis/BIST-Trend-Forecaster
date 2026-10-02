# app.py
import streamlit as st
import pandas as pd
import xgboost as xgb
import yfinance as yf
import ta
import plotly.graph_objects as go
from src import config, features
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
selected_ticker = st.sidebar.selectbox("Hisse Senedi Seçiniz", [t.replace(".IS","") for t in config.TICKERS])
selected_ticker_full = selected_ticker + ".IS"

# Model Yükleme
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(config.MODEL_PATH)
    return model

# Canlı Veri Çekme ve İşleme Fonksiyonu
def get_prediction_data(ticker):
    # Modelin indikatörleri hesaplayabilmesi için son 6 ayın verisine ihtiyacı var
    df = yf.download(ticker, period="6mo", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # DÜZELTME BURADA: 'Ticker' yerine küçük harfle 'ticker' yaptık
    df['ticker'] = ticker.replace(".IS", "")
    
    df.reset_index(inplace=True)
    
    # Sütun isimlerini düzenle (features.py 'Date' ve küçük harfli sütunlar bekliyor)
    new_columns = {}
    for col in df.columns:
        if col.lower() == 'date':
            new_columns[col] = 'Date' # Date büyük kalsın
        elif col == 'ticker':
            new_columns[col] = 'ticker' # ticker küçük kalsın
        else:
            new_columns[col] = col.lower() # open, close, high, low vs. küçük olsun
            
    df.rename(columns=new_columns, inplace=True)
    
    # Feature Engineering Scriptini Kullan
    # drop_incomplete_target=False: canlı tahminde bugünün hedefi (yarının kapanışı)
    # henüz bilinmez; bu normalde eğitimde düşürülen son günü burada tutar.
    df_processed = features.add_features(df, drop_incomplete_target=False)

    # Sadece en son günü al (Yarın için tahmin yapacağız)
    last_row = df_processed.iloc[[-1]]
    return last_row, df # df grafik çizimi için lazım

# Ana Akış
try:
    if not os.path.exists(config.MODEL_PATH):
        st.error("Model dosyası bulunamadı! Lütfen önce `src/model_train.py` çalıştırın.")
    else:
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

except Exception as e:
    st.error(f"Bir hata oluştu: {e}")