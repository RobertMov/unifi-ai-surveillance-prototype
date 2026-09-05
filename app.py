import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="UniFi Protect AI Dashboard", layout="wide")

st.title("UniFi Protect - AI Detection Prototype")
st.subheader("Modul detectie obiecte si clasificare evenimente video")

# Incarcare model YOLO
@st.cache_resource
def get_model():
    return YOLO("yolov8n.pt")

model = get_model()

# Configurare parametri
st.sidebar.header("Setari Camera")
camera_id = st.sidebar.selectbox("Camera activa", ["Cam 1 - Poarta", "Cam 2 - Parcare", "Cam 3 - Depozit"])
conf_thresh = st.sidebar.slider("Confidence threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
selected_classes = st.sidebar.multiselect("Clase urmarite", ["person", "car", "truck", "bus", "motorcycle"], default=["person", "car", "truck"])

st.sidebar.markdown("---")
st.sidebar.subheader("Info Retea")
st.sidebar.text("IP: 192.168.10.105\nVLAN: 10 (Surveillance)\nProtocol: RTSP / PoE+")

# Upload fisier video
uploaded_video = st.file_uploader("Selecteaza fisier video (mp4)", type=["mp4", "avi"])

col_video, col_logs = st.columns([2, 1])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    with col_video:
        st.write("Live Stream")
        video_placeholder = st.empty()

    with col_logs:
        st.write("Jurnal Evenimente (Logs)")
        logs_placeholder = st.empty()

    event_list = []
    frame_idx = 0

    process_btn = st.checkbox("Ruleaza procesarea", value=True)

    while cap.isOpened() and process_btn:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # Procesam la fiecare 3 cadre pentru performanta
        if frame_idx % 3 != 0:
            continue

        results = model(frame, conf=conf_thresh, verbose=False)
        frame_out = results[0].plot()

        for box in results[0].boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            confidence = float(box.conf[0])

            if name in selected_classes:
                event_list.insert(0, {
                    "Timp": datetime.now().strftime("%H:%M:%S"),
                    "Sursa": camera_id,
                    "Obiect": name,
                    "Incredere": f"{confidence:.2f}"
                })

        # Afisare cadru procesat
        frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", width="stretch")

        # Actualizare tabel evenimente
        if len(event_list) > 0:
            df = pd.DataFrame(event_list[:10])
            logs_placeholder.dataframe(df, width="stretch")

    cap.release()
else:
    st.info("Incarca un clip video pentru a rula testul.")