import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import re
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from pyvis.network import Network
import streamlit.components.v1 as components

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="SAPU TANGAN by TradeAI", layout="wide", page_icon="TradeAI_Logo.png")

# --- CSS ---
st.markdown("""
<style>
    header {visibility: hidden;}
    /* Reduce margin above the sidebar header */
    .st-emotion-cache-16txtl3 { margin-top: -75px; }
    /* Move the logo up */
    .st-emotion-cache-1v0mbdj { margin-top: -50px; }
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Roboto', sans-serif; }
    .main { background-color: #f0f2f5; }
    .st-emotion-cache-18ni7ap { background-color: #005FAC; border-bottom: 5px solid #FAB715; }
    .stButton>button { border: 2px solid #FAB715; border-radius: 5px; background-color: #FAB715; color: #005FAC; font-weight: bold; }
    .stButton>button:hover { background-color: #ffffff; color: #005FAC; border: 2px solid #005FAC; }
    .st-emotion-cache-1g6go7w { border-left: 7px solid #005FAC; padding: 1.2rem; border-radius: 8px; background-color: white; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    h1, h2, h3 { color: #005FAC; }
    .text-justify, p { color: black; }
    button[data-testid="stBaseButton-headerNoPadding"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Fungsi Bantuan & Analisis ---
@st.cache_data
def load_resources():
    resources = {}
    model_dir = 'saved_models'
    try:
        resources['vectorizer'] = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
        resources['kmeans'] = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
        resources['models_flag_1'] = joblib.load(os.path.join(model_dir, 'models_flag_1.pkl'))
        resources['models_flag_2'] = joblib.load(os.path.join(model_dir, 'models_flag_2.pkl'))
        resources['models_flag_3'] = joblib.load(os.path.join(model_dir, 'models_flag_3.pkl'))
        resources['models_flag_4'] = joblib.load(os.path.join(model_dir, 'models_flag_4.pkl'))
        resources['profil_risiko_importir'] = joblib.load(os.path.join(model_dir, 'profil_risiko_importir.pkl'))
        resources['kelompok_deskripsi'] = joblib.load(os.path.join(model_dir, 'kelompok_deskripsi_map.pkl'))
        resources['profil_risiko_pemasok'] = joblib.load(os.path.join(model_dir, 'profil_risiko_pemasok.pkl'))
        resources['profil_risiko_rute'] = joblib.load(os.path.join(model_dir, 'profil_risiko_rute.pkl'))
        df_source = pd.read_csv('./data_sintetik_v2.csv', sep=';')
        resources['df_source'] = df_source
        resources['df_sample'] = pd.read_csv('./sample_batch_1000_varied.csv', sep=';')
        resources['countries'] = sorted(df_source['negara_asal'].dropna().unique().tolist())
        resources['ports'] = sorted(df_source['pelabuhan_masuk'].dropna().unique().tolist())
        
        return resources
    except FileNotFoundError as e:
        st.sidebar.error(f"Error: {e}. Pastikan file model dan dataset tersedia.")
        return None

def bersihkan_teks(teks):
    if pd.isnull(teks): return ""
    teks = teks.lower()
    teks = re.sub(r'\d+', '', teks)
    return teks

def create_network_graph(df_anomalies, source_col, target_col):
    if df_anomalies.empty or source_col not in df_anomalies.columns or target_col not in df_anomalies.columns:
        return None
    net = Network(height="600px", width="100%", bgcolor="#f0f2f5", font_color="black", notebook=True, directed=True)
    net.set_options("""
    var options = {
      "physics": { "forceAtlas2Based": { "gravitationalConstant": -50, "centralGravity": 0.005, "springLength": 100 } }
    }
    """)
    for index, row in df_anomalies.iterrows():
        source_node = row[source_col]
        target_node = row[target_col]
        net.add_node(source_node, label=source_node, color="#005FAC", size=15, title=f"{source_col}: {source_node}")
        net.add_node(target_node, label=target_node, color="#FAB715", size=10, title=f"{target_col}: {target_node}")
        net.add_edge(source_node, target_node)
    file_path = 'network.html'
    try:
        net.save_graph(file_path)
        return file_path
    except:
        return None

def run_single_transaction_analysis(inputs):
    if not resources: return {"error": "Model tidak dimuat."}
    results = {}
    
    deskripsi_bersih = bersihkan_teks(inputs['deskripsi_barang'])
    vektor = resources['vectorizer'].transform([deskripsi_bersih])
    id_kelompok = resources['kmeans'].predict(vektor)[0]
    results['info'] = {"ID Kelompok": int(id_kelompok)}

    try:
        model_f1 = resources['models_flag_1'][id_kelompok]
        harga_scaled = model_f1['scaler'].transform([[inputs['harga_satuan']]])
        flag_if = 1 if model_f1['isolation_forest'].predict(harga_scaled)[0] == -1 else 0
        flag_iqr = 1 if not (model_f1['iqr_bounds']['lower'] <= inputs['harga_satuan'] <= model_f1['iqr_bounds']['upper']) else 0
        results['flag_1'] = {"is_anomaly": bool(flag_if and flag_iqr), "detail": f"IF: {bool(flag_if)}, IQR: {bool(flag_iqr)}"}
    except KeyError:
        results['flag_1'] = {"is_anomaly": False, "detail": "Tidak ada model untuk kelompok ini."}

    profil_importir = resources['profil_risiko_importir']
    importer_risk = profil_importir[profil_importir['importir'] == inputs['importir']]
    if not importer_risk.empty and importer_risk.iloc[0]['flag_importir_mencurigakan'] == 1:
        results['flag_5'] = {"is_risky": True, "detail": f"Importir berisiko tinggi (rasio: {importer_risk.iloc[0]['rasio_risiko']:.2f})."}
    else:
        results['flag_5'] = {"is_risky": False, "detail": "Importir tidak masuk profil berisiko."}

    profil_pemasok = resources['profil_risiko_pemasok']
    pemasok_risk = profil_pemasok[profil_pemasok['nama_pemasok'] == inputs['nama_pemasok']]
    if not pemasok_risk.empty and pemasok_risk.iloc[0]['flag_pemasok_mencurigakan'] == 1:
        results['flag_6'] = {"is_risky": True, "detail": f"Pemasok berisiko tinggi (rasio: {pemasok_risk.iloc[0]['rasio_risiko']:.2f})."}
    else:
        results['flag_6'] = {"is_risky": False, "detail": "Pemasok tidak masuk profil berisiko."}

    rute = f"{inputs['negara_asal']} -> {inputs['pelabuhan_bongkar']}"
    profil_rute = resources['profil_risiko_rute']
    route_risk = profil_rute[profil_rute['rute_perdagangan'] == rute]
    if not route_risk.empty and route_risk.iloc[0]['flag_rute_mencurigakan'] == 1:
        results['flag_7'] = {"is_risky": True, "detail": f"Rute berisiko tinggi (rasio: {route_risk.iloc[0]['rasio_risiko']:.2f})."}
    else:
        results['flag_7'] = {"is_risky": False, "detail": "Rute tidak masuk profil berisiko."}
        
    return results

def run_full_analysis(inputs):
    if not resources: return {}
    results = run_single_transaction_analysis(inputs)

    try:
        deskripsi_bersih = bersihkan_teks(inputs['uraian'])
        vektor = resources['vectorizer'].transform([deskripsi_bersih])
        id_kelompok = resources['kmeans'].predict(vektor)[0]
        model_f2 = resources['models_flag_2'][id_kelompok]
        
        rasio = inputs.get('harga_invoice', 0) / (inputs.get('netto', 1) + 1e-6)
        rasio_scaled = model_f2['scaler'].transform([[rasio]])
        is_anomaly_f2 = model_f2['isolation_forest'].predict(rasio_scaled)[0] == -1
        results['flag_2'] = {"is_anomaly": is_anomaly_f2, "detail": f"Rasio Invoice/Netto: {rasio:.2f}"}
    except (KeyError, ZeroDivisionError):
        results['flag_2'] = {"is_anomaly": False, "detail": "Tidak dapat menghitung risiko Phantom Shipping."}

    try:
       results['flag_3'] = {"is_anomaly": False, "detail": "Analisis Smurfing memerlukan pra-pemrosesan data batch."}
    except Exception:
        results['flag_3'] = {"is_anomaly": False, "detail": "Error pada analisis Smurfing."}

    try:
        results['flag_4'] = {"is_anomaly": False, "detail": "Analisis Trantib memerlukan data historis."}
    except Exception:
        results['flag_4'] = {"is_anomaly": False, "detail": "Error pada analisis Trantib."}

    return results

def run_batch_analysis(df):
    if not resources:
        st.error("Model tidak dimuat.")
        return pd.DataFrame()
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    smurfing_agg = df.groupby(['tanggal', 'importir', 'deskripsi_barang']).agg(
        freq=('importir', 'count'),
        mean_val=('nilai_invoice', 'mean'),
        std_val=('nilai_invoice', 'std')
    ).reset_index()
    smurfing_agg['std_val'] = smurfing_agg['std_val'].fillna(0)

    # --- Pra-pemrosesan untuk Flag 4 (Trantib) ---
    df_sorted = df.sort_values(by=['importir', 'deskripsi_barang', 'tanggal'])
    df_sorted['baseline_netto'] = df_sorted.groupby(['importir', 'deskripsi_barang'])['netto_barang'].expanding().mean().reset_index(level=[0,1], drop=True)
    df_sorted['deviasi_netto'] = (df_sorted['netto_barang'] - df_sorted['baseline_netto']) / (df_sorted['baseline_netto'] + 1e-6)
    
    results_list = []
    for index, row in df_sorted.iterrows():
        inputs = row.to_dict()
        analysis_result = run_full_analysis(inputs)
        
        row['flag_1'] = 1 if analysis_result.get('flag_1', {}).get('is_anomaly') else 0
        row['flag_2'] = 1 if analysis_result.get('flag_2', {}).get('is_anomaly') else 0
        
        smurfing_check = smurfing_agg[
            (smurfing_agg['tanggal'] == row['tanggal']) &
            (smurfing_agg['importir'] == row['importir']) &
            (smurfing_agg['deskripsi_barang'] == row['deskripsi_barang'])
        ]
        if not smurfing_check.empty:
            try:
                smurfing_features = smurfing_check[['freq', 'mean_val', 'std_val']].values
                model_f3 = resources['models_flag_3']
                is_anomaly_f3 = model_f3['isolation_forest'].predict(smurfing_features)[0] == -1
                row['flag_3'] = 1 if is_anomaly_f3 else 0
            except Exception:
                row['flag_3'] = 0
        else:
            row['flag_3'] = 0

        try:
            model_f4 = resources['models_flag_4']
            is_anomaly_f4 = model_f4['isolation_forest'].predict([[row['deviasi_netto']]])[0] == -1
            row['flag_4'] = 1 if is_anomaly_f4 else 0
        except Exception:
            row['flag_4'] = 0

        row['flag_5'] = 1 if analysis_result.get('flag_5', {}).get('is_risky') else 0
        row['flag_6'] = 1 if analysis_result.get('flag_6', {}).get('is_risky') else 0
        row['flag_7'] = 1 if analysis_result.get('flag_7', {}).get('is_risky') else 0
        results_list.append(row)

    return pd.DataFrame(results_list)

# --- Main App ---
resources = load_resources()

with st.sidebar:
    st.image("TradeAI_Logo.png", use_container_width=True)
    st.header("Navigasi Portal")
    app_mode = st.radio(
        "Pilih Layanan:",
        ["Home", "Analisis Batch", "Analisis Transaksional", "Profil Risiko", "Modeling"],
        label_visibility="collapsed"
    )
    st.divider()
    st.info("SAPU TANGAN v1.0")

main_container = st.container()
with main_container:
    if app_mode == "Home":
        st.title("TradeAI: Masa Depan Intelijen Keuangan Perdagangan")
        st.markdown("---")
        st.markdown("""
        <div class="text-justify">
        Perdagangan global adalah mesin ekonomi dunia, namun kompleksitasnya seringkali dieksploitasi untuk aktivitas ilegal seperti pencucian uang dan penghindaran pajak yang merugikan negara miliaran dolar setiap tahun. Metode konvensional gagal mengimbangi kecepatan dan kecanggihan para pelaku kejahatan.
        <br><br>
        Menjawab tantangan ini, Tim <b>TradeAI</b> mempersembahkan <b>SAPU TANGAN (Sistem Analisis Pencucian Uang pada Transaksi Perdagangan)</b>, sebuah <i>platform</i> kecerdasan buatan yang dirancang untuk "menyapu bersih" anomali transaksi yang tersembunyi di antara jutaan data perdagangan. Kami tidak hanya mendeteksi, kami membedah setiap transaksi untuk mengungkap pola kejahatan yang paling rumit sekalipun.
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        st.header("Arsitektur Model Deteksi Berlapis")
        st.markdown("**SAPU TANGAN** menerapkan ansambel sinergis dari model-model *machine learning* di mana setiap model dirancang untuk menargetkan fokus spesifik dari Tindak Pidana Pencucian Uang Berbasis Perdagangan (*Trade Based Money Laundering*).")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Deteksi Anomali Harga")
            st.markdown("""
            <div class="text-justify">
            <b>Dampak Ekonomi:</b> <i>Under-invoicing</i> / <i>Over-invoicing</i> mendistorsi arus modal dan menyebabkan kerugian pendapatan negara yang signifikan dari perpajakan.
            <br><br>
            <b>Pendekatan Teknis:</b> Kami menggunakan model hibrida yang mengkombinasikan <i>unsupervised clustering</i> (<i>K-Means</i>) pada deskripsi barang yang telah divektorisasi (<i>TF-IDF</i>), diikuti dengan analisis <b>Isolation Forest</b> dan <b>Interquartile Range (IQR)</b> per klaster. Ini memungkinkan model untuk secara dinamis mempelajari harga pasar dan menandai deviasi harga yang signifikan.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📦 Deteksi *Phantom Shipping*")
            st.markdown("""
            <div class="text-justify">
            <b>Dampak Ekonomi:</b> Pengiriman fiktif adalah <i>direct channel</i> untuk mencuci uang dalam jumlah besar tanpa adanya aktivitas ekonomi riil.
            <br><br>
            <b>Pendekatan Teknis:</b> Model ini berfokus pada <b>rasio invoice-vs-netto</b>, sebuah <i>feature</i> krusial untuk mendeteksi transaksi tidak logis. <b>Isolation Forest</b> dilatih pada distribusi rasio yang telah dinormalisasi per klaster yang secara efektif mengidentifikasi <i>outlier</i> di mana nilai masif dideklarasikan untuk berat fisik yang tidak masuk akal.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("💸 Deteksi *Smurfing*")
            st.markdown("""
            <div class="text-justify">
            <b>Dampak Ekonomi:</b> Memecah transaksi agar berada di bawah ambang batas pelaporan adalah teknik klasik pencucian uang untuk menghindari pengawasan regulator.
            <br><br>
            <b>Pendekatan Teknis:</b> Kami menggeser pola analisis dari transaksi tunggal ke <b>perilaku harian importir</b>. Dengan mengagregasi frekuensi, nilai rata-rata dan standar deviasi transaksi harian, <b>Isolation Forest</b> mengidentifikasi importir yang menunjukkan perilaku frekuensi tinggi pada hari tertentu dengan importasi komoditi yang sejenis.
            </div>
            <br>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📋 Deteksi Pergeseran Perilaku")
            st.markdown("""
            <div class="text-justify">
            <b>Dampak Ekonomi:</b> Perubahan mendadak pada volume pengiriman importir dapat mengindikasikan penyalahgunaan bisnis yang sah untuk tujuan terlarang.
            <br><br>
            <b>Pendekatan Teknis:</b> Model ini membangun <b>baseline historis dinamis</b> untuk setiap pasangan <code>(importir dan deskripsi barang)</code> menggunakan metode <i>expanding window mean</i>. Data historis akan menandai setiap transaksi baru di mana berat bersihnya menyimpang secara signifikan dari norma historis yang telah terbentuk.
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        st.success("Jelajahi menu di sidebar untuk menguji model-model ini melalui **Analisis Batch** atau **Analisis Transaksional**.")

    elif app_mode == "Analisis Batch":
        st.title("Layanan Analisis Batch")
        st.markdown("Unggah file CSV berisi data transaksi impor untuk dianalisis secara massal.")
        
        if 'df_sample' in resources:
            csv_sample = resources['df_sample'].to_csv(index=False, sep=';').encode('utf-8')
            st.download_button(
                label="📥 Unduh Template CSV (1000 Baris)",
                data=csv_sample,
                file_name='template_uji_coba.csv',
                mime='text/csv',
                use_container_width=True
            )

        st.divider()

        uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
        
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"Berhasil memuat **{len(df_upload)} baris**. Klik tombol di bawah untuk memulai.")
            
            if st.button("Proses File Batch", use_container_width=True):
                with st.spinner("Menganalisis seluruh data menggunakan model..."):
                    # Pastikan kolom yang dibutuhkan ada
                    required_cols = ['importir', 'nama_pemasok', 'deskripsi_barang', 'nilai_invoice', 'negara_asal', 'pelabuhan_bongkar', 'tanggal', 'netto_barang']
                    if not all(col in df_upload.columns for col in required_cols):
                        st.error(f"File yang diunggah harus berisi kolom berikut: {', '.join(required_cols)}")
                    else:
                        df_result = run_batch_analysis(df_upload)
                        st.session_state['df_result'] = df_result
            
            if 'df_result' in st.session_state:
                st.success("Analisis Batch Selesai!")
                st.divider()
                st.header("📈 Dashboard Hasil Analisis")
                df_result = st.session_state['df_result']
                
                total_f1 = df_result['flag_1'].sum()
                total_f2 = df_result['flag_2'].sum()
                total_f3 = df_result['flag_3'].sum()
                total_f4 = df_result['flag_4'].sum()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Anomali Harga", f"{total_f1} Kasus")
                col2.metric("Phantom Shipping", f"{total_f2} Kasus")
                col3.metric("Smurfing", f"{total_f3} Kasus")
                col4.metric("Anomali Trantib", f"{total_f4} Kasus")
                
                st.divider()

                viz_tab1, viz_tab2 = st.tabs(["Statistik Umum", "Grafik Jaringan Hubungan"])

                with viz_tab1:
                    st.subheader("Visualisasi Statistik Anomali")
                    
                    anomaly_counts = {
                        'Anomali Harga': total_f1,
                        'Phantom Shipping': total_f2,
                        'Smurfing': total_f3,
                        'Anomali Trantib': total_f4
                    }
                    df_anomaly_dist = pd.DataFrame(list(anomaly_counts.items()), columns=['Jenis Anomali', 'Jumlah Kasus'])
                    fig_pie = px.pie(df_anomaly_dist, names='Jenis Anomali', values='Jumlah Kasus',
                                     title='Proporsi Jenis Anomali yang Terdeteksi',
                                     color_discrete_sequence=px.colors.sequential.OrRd_r)
                    st.plotly_chart(fig_pie, use_container_width=True)

                    st.divider()

                    col_viz1, col_viz2 = st.columns(2)
                    with col_viz1:
                        st.markdown("##### Top 10 Importir Berisiko")
                        top_importir = df_result[(df_result['flag_1'] == 1) | (df_result.get('flag_2', 0) == 1) | (df_result.get('flag_4', 0) == 1)]['importir'].value_counts().nlargest(10)
                        st.bar_chart(top_importir)
                    
                    with col_viz2:
                        st.markdown("##### Top 10 Negara Asal Berisiko")
                        top_countries = df_result[(df_result['flag_1'] == 1) | (df_result.get('flag_2', 0) == 1) | (df_result.get('flag_4', 0) == 1)]['negara_asal'].value_counts().nlargest(10)
                        st.bar_chart(top_countries)

                with viz_tab2:
                    st.subheader("Jaringan Importir dan Pemasok Terindikasi Anomali")
                    df_anomalies_for_graph = df_result[(df_result['flag_1'] == 1) | (df_result.get('flag_2', 0) == 1) | (df_result.get('flag_4', 0) == 1)]
                    
                    if 'nama_pemasok' in df_anomalies_for_graph.columns:
                        with st.spinner("Membangun grafik jaringan..."):
                            html_path = create_network_graph(df_anomalies_for_graph, 'nama_pemasok', 'importir')
                            if html_path:
                                with open(html_path, 'r', encoding='utf-8') as f:
                                    source_code = f.read()
                                components.html(source_code, height=610, scrolling=True)
                            else:
                                st.info("Tidak ada data yang cukup untuk membuat grafik jaringan.")
                    else:
                        st.warning("Kolom 'nama_pemasok' tidak ditemukan dalam data yang diunggah untuk membuat grafik jaringan.")

                st.divider()
                st.subheader("Tabel Rincian Transaksi Anomali")
                df_anomalies_only = df_result[(df_result['flag_1'] == 1) | (df_result.get('flag_2', 0) == 1) | (df_result.get('flag_4', 0) == 1)]
                st.dataframe(df_anomalies_only)

    elif app_mode == "Analisis Transaksional":
        st.title("Layanan Analisis Transaksional")
        st.markdown("Masukkan detail transaksi tunggal untuk mendapatkan penilaian risiko instan.")
        if resources:
            with st.form("transactional_form"):
                col1, col2 = st.columns(2)
                with col1:
                    importir_input = st.text_input("Nama Importir")
                    pemasok_input = st.text_input("Nama Pemasok")
                    uraian_input = st.text_area("Deskripsi Barang")
                with col2:
                    harga_input = st.number_input("Harga Satuan", min_value=0.01, format="%.2f")
                    negara_input = st.selectbox("Negara Asal", options=resources['countries'])
                    pelabuhan_input = st.selectbox("Pelabuhan Tujuan", options=resources['ports'])
                
                submit_button = st.form_submit_button(label="Analisis Transaksi", use_container_width=True)

            if submit_button:
                if not all([importir_input, pemasok_input, uraian_input, harga_input]):
                    st.warning("Harap isi semua field yang tersedia.")
                else:
                    inputs = {
                        "importir": importir_input,
                        "nama_pemasok": pemasok_input,
                        "deskripsi_barang": uraian_input,
                        "harga_satuan": harga_input,
                        "negara_asal": negara_input,
                        "pelabuhan_bongkar": pelabuhan_input
                    }
                    with st.spinner("Menganalisis transaksi..."):
                        analysis_result = run_single_transaction_analysis(inputs)
                    
                    st.divider()
                    st.header("Hasil Penilaian Risiko")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Penilaian Risiko")
                        res_f1 = analysis_result.get('flag_1', {})
                        st.metric(
                            label="Risiko Anomali Harga",
                            value="🔴 TINGGI" if res_f1.get("is_anomaly") else "🟢 Normal",
                            help=res_f1.get("detail")
                        )
                        res_f5 = analysis_result.get('flag_5', {})
                        st.metric(
                            label="Risiko Importir",
                            value="🔴 TINGGI" if res_f5.get("is_risky") else "🟢 Normal",
                            help=res_f5.get("detail")
                        )
                        res_f6 = analysis_result.get('flag_6', {})
                        st.metric(
                            label="Risiko Pemasok",
                            value="🔴 TINGGI" if res_f6.get("is_risky") else "🟢 Normal",
                            help=res_f6.get("detail")
                        )
                        res_f7 = analysis_result.get('flag_7', {})
                        st.metric(
                            label="Risiko Rute",
                            value="🔴 TINGGI" if res_f7.get("is_risky") else "🟢 Normal",
                            help=res_f7.get("detail")
                        )
                    
                    with col2:
                        st.subheader("Prediksi Kelompok Barang")
                        kelompok_id = analysis_result.get('info', {}).get('ID Kelompok', 'N/A')
                        st.info(f"**ID Kelompok:** {kelompok_id}")
                        st.json({"input": inputs, "output": analysis_result})
        else:
            st.error("Sumber daya (model/data) tidak dapat dimuat.")

    elif app_mode == "Profil Risiko":
        st.title("🔍 Pemantauan Profil Risiko")
        st.markdown("Halaman ini menampilkan entitas (Importir, Pemasok, dan Rute Perdagangan) yang teridentifikasi memiliki risiko tinggi berdasarkan analisis historis.")

        if resources:
            RISIKO_THRESHOLD_IMPORTIR = 0.40
            MIN_TRANSAKSI_IMPORTIR = 5
            RISIKO_THRESHOLD_PEMASOK = 0.40
            MIN_TRANSAKSI_PEMASOK = 5
            RISIKO_RUTE_THRESHOLD = 0.20
            MIN_TRANSAKSI_RUTE = 5

            tab1, tab2, tab3 = st.tabs(["Risiko Importir", "Risiko Pemasok", "Risiko Rute"])

            with tab1:
                st.subheader("Profil Risiko Importir")
                st.info(f"Ambang Batas: Rasio Risiko >= {RISIKO_THRESHOLD_IMPORTIR:.0%} dan Total Transaksi >= {MIN_TRANSAKSI_IMPORTIR}")
                
                df_risk_importir = resources['profil_risiko_importir']
                df_mencurigakan_importir = df_risk_importir[df_risk_importir['flag_importir_mencurigakan'] == 1].sort_values(by='rasio_risiko', ascending=False)
                
                st.metric("Jumlah Importir Berisiko Tinggi", len(df_mencurigakan_importir))
                st.dataframe(df_mencurigakan_importir, use_container_width=True)
                
                st.subheader("Jaringan 5 Importir Paling Berisiko dan Pemasok Terkait")
                df_top5_importir = df_mencurigakan_importir.head(5)
                
                if not df_top5_importir.empty:
                    top_importir_names = df_top5_importir['importir'].tolist()
                    df_source = resources['df_source']
                    df_filtered = df_source[df_source['importir'].isin(top_importir_names)]
                    
                    with st.spinner("Membangun grafik jaringan..."):
                        html_path = create_network_graph(df_filtered, 'nama_pemasok', 'importir')
                        if html_path:
                            with open(html_path, 'r', encoding='utf-8') as f:
                                source_code = f.read()
                            components.html(source_code, height=610, scrolling=True)
                        else:
                            st.info("Tidak ada data yang cukup untuk membuat grafik jaringan.")
                else:
                    st.info("Tidak ada data importir berisiko untuk divisualisasikan.")

            with tab2:
                st.subheader("Profil Risiko Pemasok")
                st.info(f"Ambang Batas: Rasio Risiko >= {RISIKO_THRESHOLD_PEMASOK:.0%} dan Total Transaksi >= {MIN_TRANSAKSI_PEMASOK}")

                df_risk_pemasok = resources['profil_risiko_pemasok']
                df_mencurigakan_pemasok = df_risk_pemasok[df_risk_pemasok['flag_pemasok_mencurigakan'] == 1].sort_values(by='rasio_risiko', ascending=False)

                st.metric("Jumlah Pemasok Berisiko Tinggi", len(df_mencurigakan_pemasok))
                st.dataframe(df_mencurigakan_pemasok, use_container_width=True)
                
                st.subheader("Jaringan 5 Pemasok Paling Berisiko dan Importir Terkait")
                df_top5_pemasok = df_mencurigakan_pemasok.head(5)

                if not df_top5_pemasok.empty:
                    top_pemasok_names = df_top5_pemasok['nama_pemasok'].tolist()
                    df_source = resources['df_source']
                    df_filtered = df_source[df_source['nama_pemasok'].isin(top_pemasok_names)]
                    
                    with st.spinner("Membangun grafik jaringan..."):
                        html_path = create_network_graph(df_filtered, 'nama_pemasok', 'importir')
                        if html_path:
                            with open(html_path, 'r', encoding='utf-8') as f:
                                source_code = f.read()
                            components.html(source_code, height=610, scrolling=True)
                        else:
                            st.info("Tidak ada data yang cukup untuk membuat grafik jaringan.")
                else:
                    st.info("Tidak ada data pemasok berisiko untuk divisualisasikan.")

            with tab3:
                st.subheader("Profil Risiko Rute Perdagangan")
                st.info(f"Ambang Batas: Rasio Risiko >= {RISIKO_RUTE_THRESHOLD:.0%} dan Total Transaksi >= {MIN_TRANSAKSI_RUTE}")
                
                df_risk_rute = resources['profil_risiko_rute']
                df_mencurigakan_rute = df_risk_rute[df_risk_rute['flag_rute_mencurigakan'] == 1].sort_values(by='rasio_risiko', ascending=False)
                
                st.metric("Jumlah Rute Berisiko Tinggi", len(df_mencurigakan_rute))
                st.dataframe(df_mencurigakan_rute, use_container_width=True)
                
                st.subheader("Jaringan Pelabuhan Berisiko dan Importir Terkait")
                top_rutes = df_mencurigakan_rute.head(10)['rute_perdagangan'].tolist()
                top_pelabuhan = list(set([r.split(' -> ')[1] for r in top_rutes]))
                df_source = resources['df_source']
                df_filtered_pelabuhan = df_source[df_source['pelabuhan_masuk'].isin(top_pelabuhan)]
                st.warning("Untuk optimalisasi, grafik di bawah ini hanya menampilkan sampel 50 koneksi paling sering.")
                with st.spinner("Membangun grafik jaringan pelabuhan..."):
                    html_path = create_network_graph(df_filtered_pelabuhan.head(50), 'pelabuhan_masuk', 'importir')
                    if html_path:
                        with open(html_path, 'r', encoding='utf-8') as f:
                            source_code = f.read()
                        components.html(source_code, height=610, scrolling=True)
                    else:
                        st.info("Tidak ada data yang cukup untuk membuat grafik jaringan.")

                st.subheader("Jaringan Negara Asal Berisiko dan Pemasok Terkait")
                top_negara = list(set([r.split(' -> ')[0] for r in top_rutes]))
                df_filtered_negara = df_source[df_source['negara_asal'].isin(top_negara)]
                st.warning("Untuk optimalisasi, grafik di bawah ini hanya menampilkan sampel 50 koneksi paling sering.")
                with st.spinner("Membangun grafik jaringan negara asal..."):
                    html_path = create_network_graph(df_filtered_negara.head(50), 'negara_asal', 'nama_pemasok')
                    if html_path:
                        with open(html_path, 'r', encoding='utf-8') as f:
                            source_code = f.read()
                        components.html(source_code, height=610, scrolling=True)
                    else:
                        st.info("Tidak ada data yang cukup untuk membuat grafik jaringan.")

        else:
            st.error("Sumber daya profil risiko tidak dapat dimuat.")

    elif app_mode == "Modeling":
        st.title("🔬 Visualisasi Hasil Pemodelan")
        st.markdown("Halaman ini menampilkan visualisasi kunci yang dihasilkan selama proses analisis dan pemodelan data.")
        st.info("Semua chart di bawah ini interaktif. Anda bisa zoom, pan, dan hover untuk melihat detail.")
        
        chart_dir = "charts"

        @st.cache_data
        def load_chart_html(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except FileNotFoundError:
                return None

        def show_chart(file_name, title, caption):
            st.subheader(title)
            st.markdown(caption, unsafe_allow_html=True)
            html_chart = load_chart_html(os.path.join(chart_dir, file_name))
            if html_chart:
                components.html(html_chart, height=500, scrolling=True)
            else:
                st.warning(f"File chart '{file_name}' tidak ditemukan.")

        st.divider()
        st.header("📈 Analisis Anomali Harga")
        show_chart("anomali_harga_scatterplot.html", "Validasi Klaster Harga", 
                   "Visualisasi ini memvalidasi efektivitas *K-Means clustering* dalam mengelompokkan produk serupa. Titik data merah (anomali) yang berada jauh dari pusat massa *klaster* birunya menunjukkan bahwa model *Isolation Forest* berhasil mengidentifikasi deviasi harga signifikan dalam konteks semantik yang relevan. Ini adalah bukti kuantitatif kemampuan model untuk memahami harga wajar per kelompok produk.")
        show_chart("anomali_harga_boxplot.html", "Distribusi Harga Antar Klaster",
                   "*Box plot* ini memberikan pandangan statistik tentang variasi harga di dalam dan antar *klaster* produk. *Klaster* dengan rentang *interkuartil* (IQR) yang sempit menandakan komoditas dengan harga stabil, sementara *klaster* dengan rentang lebar dan banyak *outlier* menunjukkan pasar yang lebih *volatil* atau *heterogen*. Analisis ini krusial untuk mengkalibrasi sensitivitas deteksi anomali per jenis barang.")
        show_chart("anomali_harga_barchart.html", "Peringkat Klaster Berisiko",
                   "*Chart* ini mengidentifikasi kelompok-kelompok komoditas yang paling sering dieksploitasi untuk *mis-invoicing*. Dari perspektif ekonomi, ini memungkinkan regulator untuk memfokuskan audit pada sektor-sektor paling rentan, mengoptimalkan alokasi sumber daya investigasi dan secara proaktif mencegah kerugian negara.")

        st.divider()
        st.header("📦 Analisis *Phantom Shipping*")
        show_chart("phantom_histogram.html", "Distribusi Rasio Invoice-To-Netto",
                   "*Histogram* ini menunjukkan distribusi fitur kunci: rasio nilai *invoice* terhadap berat bersih. Ekor panjang (*long tail*) pada distribusi ini secara inheren mencurigakan, merepresentasikan transaksi di mana nilai ekonomi yang sangat besar dikaitkan dengan massa fisik yang kecil. Garis merah menandai ambang batas anomali yang dihitung secara statistik, memisahkan transaksi wajar dari yang berpotensi fiktif.")
        show_chart("phantom_scatterplot.html", "Identifikasi *Outlier* Nilai Transaksi vs Berat",
                   "Setiap titik pada *scatter plot* ini adalah sebuah transaksi. Sumbu *logaritmik* digunakan untuk menangani rentang nilai yang ekstrim. Titik-titik merah yang terisolasi adalah *red flag* yang jelas: transaksi dengan nilai *invoice* sangat tinggi namun berat bersih sangat rendah, sebuah karakteristik utama dari *phantom shipping* yang digunakan untuk memindahkan dana secara ilegal.")

        st.divider()
        st.header("💸 Analisis *Smurfing*")
        show_chart("smurfing_scatterplot.html", "Pemetaan Perilaku Transaksi Harian",
                   "*Plot* ini menggeser fokus dari transaksi tunggal ke perilaku agregat harian per importir. Sumbu-sumbunya merepresentasikan frekuensi dan nilai rata-rata transaksi. Pola ini secara efektif menghindari ambang batas pelaporan regulator dan sulit dideteksi tanpa analisis agregat.")
        show_chart("smurfing_waktu.html", "Analisis Tren Temporal Anomali",
                   "Grafik ini melacak frekuensi berbagai jenis anomali dari waktu ke waktu. Lonjakan pada tipe anomali tertentu dapat berkorelasi dengan perubahan kebijakan ekonomi, tarif impor baru atau bahkan peristiwa geopolitik. Ini memberikan wawasan makroekonomi tentang bagaimana pelaku kejahatan beradaptasi dan mengeksploitasi sistem perdagangan global.")

        st.divider()
        st.header("📋 Analisis Pergeseran Perilaku")
        show_chart("trantib_chart.html", "Studi Kasus: Deviasi dari Baseline Historis",
                   "Visualisasi ini adalah studi kasus pada satu importir yang menunjukkan perubahan perilaku drastis. Garis merah adalah *baseline* historis volume impor (rata-rata bergerak), sementara garis biru adalah volume aktual. Titik merah menandai transaksi di mana volume impor tiba-tiba melonjak jauh di atas norma historisnya, mengindikasikan potensi penyalahgunaan entitas bisnis yang sah untuk aktivitas ilegal.")

        st.divider()
        st.header("🚢 Profil Risiko Importir")
        show_chart("importir_stacked.html", "Dekomposisi Risiko per Importir", 
                   "Bukan hanya memberi skor risiko tunggal, *chart* ini membedah *DNA* anomali untuk setiap importir berisiko tinggi. Apakah mereka lebih sering melakukan *mis-invoicing*, *phantom shipping* atau *smurfing*? Pemahaman ini memungkinkan investigasi yang jauh lebih terarah dan efisien.")
        show_chart("importir_scatter.html", "Matriks Risiko Volume Importir",
                   "*Plot* ini memetakan seluruh populasi importir berdasarkan volume transaksi (sumbu x) dan rasio anomali (sumbu y). Importir yang berwarna merah adalah target prioritas tertinggi: mereka tidak hanya sering melakukan transaksi tetapi sebagian besar transaksi mereka mencurigakan. Ini adalah alat strategis untuk alokasi sumber daya audit.")

        st.divider()
        st.header("🚚 Profil Risiko Pemasok")
        show_chart("pemasok_stacked.html", "Dekomposisi Risiko per Pemasok",
                   "Serupa dengan profil importir, visualisasi ini menganalisis pemasok mana yang paling sering terlibat dalam transaksi anomali. Mengidentifikasi pemasok berisiko tinggi di luar negeri adalah langkah penting dalam membongkar jaringan pencucian uang transnasional.")
        show_chart("pemasok_scatter.html", "Matriks Risiko Volume Pemasok",
                   "Matriks ini membantu mengidentifikasi pemasok asing yang mungkin merupakan *pabrik dokumen palsu* atau *entitas cangkang (shell company)*. Pemasok yang berwarna merah memiliki volume tinggi dan rasio anomali tinggi, kemungkinan besar adalah fasilitator utama dalam skema Trade Based Money Laundering.")

        st.divider()
        st.header("🌐 Profil Risiko Rute")
        show_chart("rute_stacked.html", "Dekomposisi Risiko per Rute Perdagangan",
                   "Analisis ini mengungkap rute perdagangan (negara asal -> pelabuhan tujuan) yang paling sering dieksploitasi. Informasi ini sangat berharga untuk intelijen Bea Cukai dan lembaga penegak hukum lainnya dan memungkinkan mereka untuk meningkatkan pengawasan pada koridor perdagangan spesifik yang terbukti rentan.")
        show_chart("rute_scatter.html", "Matriks Risiko Volume Rute",
                   "Dengan memetakan semua rute perdagangan, kita dapat secara strategis mengidentifikasi *jalur sutra modern* untuk pencucian uang. Rute perdagangan yang berwarna merah tidak hanya memiliki volume perdagangan yang tinggi tetapi juga tingkat anomali yang tidak proporsional, menjadikannya subjek vital untuk kerja sama penegakan hukum internasional dan antar lembaga.")
