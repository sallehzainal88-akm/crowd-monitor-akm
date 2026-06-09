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

# Memuatkan model AI YOLOv8 Medium secara selamat di server Streamlit
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

# Cipta 'Sistem Barisan Data' (Queue) untuk menghantar data keluar dari utas video
if "data_queue" not in st.session_state:
    st.session_state.data_queue = queue.Queue()

# Tempat simpan senarai data laporan untuk dijadikan fail CSV
if "rekod_laporan_list" not in st.session_state:
    st.session_state.rekod_laporan_list = []
if "waktu_mula_sesi" not in st.session_state:
    st.session_state.waktu_mula_sesi = None

class CrowdVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.senarai_id_pelawat = set()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        tinggi, lebar, _ = img.shape

        # Memaksa YOLOv8 menggunakan penjejak ByteTrack kalis ralat 'lap'
        hasil_ai_list = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)
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

        # Hantar data bilangan terkini ke luar utas video menggunakan Queue
        st.session_state.data_queue.put((dalam_frame_sekarang, jumlah_pelawat_hari_ini))

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
    
    # =========================================================================
    # PENYELESAIAN UTAMA: MENAMBAH SENARAI PELAYAN STUN GLOBAL (VERSI SINTAKSIS BETUL)
    # =========================================================================
    konfigurasi_rtc = {
        "iceServers": [
            {
                "urls": [
                    "stun:://google.com",
                    "stun:://google.com",
                    "stun:://google.com",
                    "stun:stun.stunprotocol.org:3478"
                ]
            }
        ]
    }
    
    ctx = webrtc_streamer(
        key="crowd-monitor",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=CrowdVideoProcessor,
        async_processing=True,
        rtc_configuration=konfigurasi_rtc, # Memasukkan tetapan rangkaian baharu
    )
    # =========================================================================

with col2:
    st.subheader("Statistik Pengunjung Semasa")
    petak_data_semasa = st.empty()
    petak_data_jumlah = st.empty()
    
    petak_data_semasa.metric(label="Dalam Frame Sekarang", value="0 orang")
    petak_data_jumlah.metric(label="Jumlah Pelawat Unik Hari Ini", value="0 orang")
    
    st.write("---")
    st.subheader("Pusat Muat Turun Laporan")
    petak_butang_download = st.empty()

# Rekod waktu mula sebaik sahaja pengguna mengklik butang "Start" pada kamera
if ctx.state.playing and st.session_state.waktu_mula_sesi is None:
    st.session_state.waktu_mula_sesi = datetime.now().strftime("%H:%M:%S")

# Gelung pengawasan utama untuk menangkap data dari utas video dan memaparkannya di web
while ctx.state.playing:
    try:
        dalam_frame, jumlah_pelawat = st.session_state.data_queue.get(timeout=0.1)
        petak_data_semasa.metric(label="Dalam Frame Sekarang", value=f"{dalam_frame} orang")
        petak_data_jumlah.metric(label="Jumlah Pelawat Unik Hari Ini", value=f"{jumlah_pelawat} orang")
        st.session_state.jumlah_akhir_pengunjung = jumlah_pelawat
    except queue.Empty:
        continue

if not ctx.state.playing and st.session_state.waktu_mula_sesi is None:
    petak_butang_download.info("Sila klik butang 'Start' pada kamera untuk memulakan sesi rekod.")

# Apabila sesi kamera bertukar daripada hidup kepada BERHENTI (User klik Stop)
if not ctx.state.playing and st.session_state.waktu_mula_sesi is not None:
    waktu_tamat = datetime.now().strftime("%H:%M:%S")
    tarikh_hari_ini = datetime.now().strftime("%Y-%m-%d")
    jumlah_total = st.session_state.get("jumlah_akhir_pengunjung", 0)
    
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
    
    st.session_state.waktu_mula_sesi = None
