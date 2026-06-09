import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import cv2
from ultralytics import YOLO
import queue
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Crowd Monitor AI", layout="wide")
st.title("Crowd Monitor AI - Team IT AKM")
st.write("Sistem pengesan pengunjung masa-nyata berasaskan kecerdasan buatan.")

# 1. Memuatkan model AI YOLOv8 Medium secara selamat
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

# 2. Cipta Queue global untuk komunikasi antara thread WebRTC & Streamlit
if "data_queue" not in st.session_state:
    st.session_state.data_queue = queue.Queue()

if "rekod_laporan_list" not in st.session_state:
    st.session_state.rekod_laporan_list = []
if "waktu_mula_sesi" not in st.session_state:
    st.session_state.waktu_mula_sesi = None
if "jumlah_akhir_pengunjung" not in st.session_state:
    st.session_state.jumlah_akhir_pengunjung = 0

# 3. Kelas Pemprosesan Video (WebRTC Thread)
class CrowdVideoProcessor(VideoProcessorBase):
    def __init__(self, data_queue):
        self.senarai_id_pelawat = set()
        self.data_queue = data_queue  # Guna queue yang di-pass, bukan st.session_state secara langsung

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        tinggi, lebar, _ = img.shape

        # Tambah classes=[0] untuk mengehadkan pengesanan kepada MANUSIA sahaja
        hasil_ai_list = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False, classes=[0])
        dalam_frame_sekarang = 0

        if len(hasil_ai_list) > 0:
            hasil_ai = hasil_ai_list[0]

            if hasattr(hasil_ai, "boxes") and hasil_ai.boxes is not None and hasil_ai.boxes.id is not None:
                kotak_objek = hasil_ai.boxes.xyxy.int().cpu().tolist()  
                id_objek = hasil_ai.boxes.id.int().cpu().tolist()       
                tahap_tepat = hasil_ai.boxes.conf.cpu().tolist()      
                dalam_frame_sekarang = len(id_objek)

                for kotak, id_orang, conf in zip(kotak_objek, id_objek, tahap_tepat):
                    x1, y1, x2, y2 = kotak
                    self.senarai_id_pelawat.add(id_orang)

                    # Lukis kotak biru pada objek
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label_teks = f"#{id_orang} [{int(conf * 100)}%]"
                    cv2.putText(img, label_teks, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        jumlah_pelawat_hari_ini = len(self.senarai_id_pelawat)

        # Masukkan data ke dalam queue dengan selamat
        self.data_queue.put((dalam_frame_sekarang, jumlah_pelawat_hari_ini))

        # Dashboard maklumat teks atas skrin kamera
        cv2.rectangle(img, (20, 20), (450, 150), (0, 0, 0), -1)
        cv2.putText(img, "CROWD MONITOR LIVE", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Dalam Frame Semasa: {dalam_frame_sekarang} orang", (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, f"JUMLAH PELAWAT UNIK: {jumlah_pelawat_hari_ini}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # Watermark Team IT AKM
        teks_watermark = "Created By Team IT AKM"
        (saiz_teks, _) = cv2.getTextSize(teks_watermark, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        posisi_x = lebar - saiz_teks - 20
        posisi_y = tinggi - 25
        cv2.putText(img, teks_watermark, (posisi_x, posisi_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        return frame.from_ndarray(img, format="bgr24")

# Set susunan halaman kepada 2 lajur seimbang
col1, col2 = st.columns(2)

with col1:
    st.subheader("Suapan Kamera Langsung")
    
    # PEMBETULAN: Format sintaks STUN Server yang betul untuk Google & Twilio public
    konfigurasi_rtc = {
        "iceServers": [
            {"urls": ["stun:://google.com"]},
            {"urls": ["stun:://google.com"]},
            {"urls": ["stun:stun.stunprotocol.org:3478"]}
        ]
    }
    
    ctx = webrtc_streamer(
        key="crowd-monitor",
        mode=WebRtcMode.SENDRECV,
        # Hantar data_queue masuk ke dalam parameter factory
        video_processor_factory=lambda: CrowdVideoProcessor(st.session_state.data_queue),
        async_processing=True,
        rtc_configuration=konfigurasi_rtc,
    )

with col2:
    st.subheader("Statistik Pengunjung Semasa")
    petak_data_semasa = st.empty()
    petak_data_jumlah = st.empty()
    
    st.write("---")
    st.subheader("Pusat Muat Turun Laporan")
    petak_butang_download = st.empty()

# 4. Pengurusan Logik State & Pengeluaran Data secara Selamat
if ctx.state.playing:
    if st.session_state.waktu_mula_sesi is None:
        st.session_state.waktu_mula_sesi = datetime.now().strftime("%H:%M:%S")

    # Baca data terakhir yang ada di dalam queue tanpa menggunakan loop "while True" yang tiada henti
    dalam_frame, jumlah_pelawat = 0, st.session_state.jumlah_akhir_pengunjung
    while not st.session_state.data_queue.empty():
        dalam_frame, jumlah_pelawat = st.session_state.data_queue.get()
    
    st.session_state.jumlah_akhir_pengunjung = jumlah_pelawat

    petak_data_semasa.metric(label="Dalam Frame Sekarang", value=f"{dalam_frame} orang")
    petak_data_jumlah.metric(label="Jumlah Pelawat Unik Hari Ini", value=f"{jumlah_pelawat} orang")
    
    # Paksa Streamlit buat refresh UI pendek secara auto jika kamera sedang aktif
    st.button("🔄 Kemas Kini Metrik Manual", key="refresh_btn")

else:
    petak_data_semasa.metric(label="Dalam Frame Sekarang", value="0 orang")
    petak_data_jumlah.metric(label="Jumlah Pelawat Unik Hari Ini", value=f"{st.session_state.jumlah_akhir_pengunjung} orang")

    if st.session_state.waktu_mula_sesi is None:
        petak_butang_download.info("Sila klik butang 'Start' pada kamera untuk memulakan sesi rekod.")

# Apabila pengguna menekan butang 'Stop'
if not ctx.state.playing and st.session_state.waktu_mula_sesi is not None:
    waktu_tamat = datetime.now().strftime("%H:%M:%S")
    tarikh_hari_ini = datetime.now().strftime("%Y-%m-%d")
    jumlah_total = st.session_state.jumlah_akhir_pengunjung
    
    data_sesi_ini = {
        "Tarikh Rekod": tarikh_hari_ini,
        "Waktu Mula Rekod": st.session_state.waktu_mula_sesi,
        "Waktu Akhir Rekod": waktu_tamat,
        "Jumlah Pengunjung": jumlah_total
    }
    
    if data_sesi_ini not in st.session_state.rekod_laporan_list:
        st.session_state.rekod_laporan_list.append(data_sesi_ini)
    
    df_laporan = pd.DataFrame(st.session_state.rekod_laporan_list)
    csv_data = df_laporan.to_csv(index=False).encode('utf-8')
    
    with col2:
        petak_butang_download.download_button(
            label="📥 Muat Turun Laporan CSV",
            data=csv_data,
            file_name=f"laporan_pengunjung_{tarikh_hari_ini}.csv",
            mime="text/csv",
            key="download-csv-btn"
        )
        st.success(f"Sesi tamat pada pukul {waktu_tamat}. Fail laporan sedia untuk dimuat turun!")
    
    # Reset waktu mula untuk sesi akan datang
    st.session_state.waktu_mula_sesi = None
