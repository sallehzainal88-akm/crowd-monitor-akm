import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="Crowd Monitor AI", layout="wide")
st.title("Crowd Monitor AI - Team IT AKM")
st.write("Sistem pengesan pengunjung masa-nyata berasaskan kecerdasan buatan.")

# KEMASKINI UTAMA: Mengarahkan pelayan Streamlit memuat turun model rasmi secara automatik
@st.cache_resource
def load_model():
    # Sistem akan auto-download fail yolov8m.pt dari server rasmi Ultralytics ke cloud Streamlit
    return YOLO("yolov8m.pt")

model = load_model()

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.senarai_id_pelawat = set()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        tinggi, lebar, _ = img.shape

        hasil_ai_list = model.track(img, persist=True, verbose=False)
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

                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label_teks = f"#{id_orang} [{int(conf * 100)}%]"
                    cv2.putText(img, label_teks, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        jumlah_pelawat_hari_ini = len(self.senarai_id_pelawat)

        # Dashboard maklumat teks atas skrin
        cv2.rectangle(img, (20, 20), (450, 150), (0, 0, 0), -1)
        cv2.putText(img, "CROWD MONITOR LIVE", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Dalam Frame Semasa: {dalam_frame_sekarang} orang", (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, f"JUMLAH PELAWAT UNIK: {jumlah_pelawat_hari_ini}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # Watermark Team IT AKM
        teks_watermark = "Created By Team IT AKM"
        (saiz_teks, _) = cv2.getTextSize(teks_watermark, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        posisi_x = lebar - saiz_teks[0] - 20
        posisi_y = tinggi - 25
        cv2.putText(img, teks_watermark, (posisi_x, posisi_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        return frame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="crowd-monitor",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=VideoProcessor,
    async_transform=True,
)
