import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="UniFi Protect AI Dashboard", layout="wide")

st.title("🛡️ Ubiquiti UniFi Protect - AI Detection Prototype")
st.caption("Prototip software pentru monitorizarea perimetrului și clasificare evenimente AI")

# Inițializare model YOLOv8 (detectează persoane, mașini etc.)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Sidebar: Configurare scenariu UniFi
st.sidebar.header("⚙️ Configurare Cameră UniFi")
camera_name = st.sidebar.selectbox("Selectează Camera", ["Cam-01: Poartă Acces", "Cam-02: Parcare Personal", "Cam-03: Depozit"])
confidence_threshold = st.sidebar.slider("Prag Încredere Detecție AI", 0.1, 1.0, 0.45)
target_classes = st.sidebar.multiselect("Filtru Obiecte AI", ["person", "car", "truck", "motorcycle"], default=["person", "car"])

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Detalii Rețea Simulare")
st.sidebar.text(f"IP Cameră: 192.168.10.105\nVLAN: 10 (Surveillance)\nStatus: Conectat (PoE+)")

# Upload sau selectare video demo
uploaded_file = st.file_uploader("Încarcă flux video pentru analiză (MP4 / AVI)", type=["mp4", "avi", "mov"])

col1, col2 = st.columns([2, 1])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    with col1:
        st.subheader("📺 Flux Video în Timp Real (RTSP AI Stream)")
        frame_placeholder = st.empty()

    with col2:
        st.subheader("🚨 Jurnal Evenimente AI Protect")
        events_placeholder = st.empty()

    events = []
    frame_count = 0

    run_sim = st.checkbox("Pornește Procesarea Fluxului", value=True)

    while cap.isOpened() and run_sim:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Procesăm câte 1 cadru la fiecare 3 pentru fluiditate
        if frame_count % 3 != 0:
            continue

        results = model(frame, conf=confidence_threshold, verbose=False)
        annotated_frame = results[0].plot()

        # Înregistrăm detecțiile
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            if label in target_classes:
                events.insert(0, {
                    "Ora": datetime.now().strftime("%H:%M:%S"),
                    "Cameră": camera_name,
                    "Eveniment": f"{label.upper()} detectat",
                    "Încredere": f"{conf:.2f}"
                })

        # Redare video în dashboard
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Actualizare tabel evenimente (ultimele 8)
        if events:
            df_events = pd.DataFrame(events[:8])
            events_placeholder.dataframe(df_events, use_container_width=True)

    cap.release()
else:
    st.info("Încarcă un clip scurt cu trafic sau pietoni pentru a începe simularea detecției.")