from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd
import torch

st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None


# ═══════════════════════════════════════════════════════════════
# ESTILOS — NIGHT VISION DARK
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {
        --bg: #080b14;
        --surface: #10141f;
        --surface-2: #161b29;
        --surface-3: #1c2233;
        --border: rgba(148, 163, 184, 0.12);
        --border-2: rgba(148, 163, 184, 0.20);
        --text: #e8ecf6;
        --text-2: #b8c1d6;
        --muted: #7d879c;
        --muted-2: #5a6378;
        --accent: #22d3ee;
        --accent-2: #10b981;
        --accent-3: #8b5cf6;
        --accent-4: #f472b6;
        --glow-cyan: rgba(34, 211, 238, 0.35);
        --glow-green: rgba(16, 185, 129, 0.35);
    }

    html, body, [class*="css"], .stApp, button, input, textarea, select {
        font-family: 'Inter', sans-serif !important;
        color: var(--text) !important;
    }
    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em !important;
    }

    /* ═══ FONDO ═══ */
    .stApp {
        background-color: var(--bg) !important;
        background-image:
            radial-gradient(ellipse at top left,    rgba(34, 211, 238, 0.10), transparent 45%),
            radial-gradient(ellipse at top right,   rgba(139, 92, 246, 0.10), transparent 45%),
            radial-gradient(ellipse at bottom,      rgba(16, 185, 129, 0.07), transparent 55%),
            linear-gradient(rgba(148, 163, 184, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(148, 163, 184, 0.035) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 100% 100%, 44px 44px, 44px 44px;
        background-attachment: fixed;
    }

    /* ═══ SIDEBAR ═══ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e1a 0%, #0d1220 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    .sb-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text);
        margin: 0.25rem 0 1.25rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sb-section {
        font-size: 0.7rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 700;
        color: var(--accent);
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
    }
    .sb-hint {
        background: rgba(34, 211, 238, 0.06);
        border-left: 3px solid var(--accent);
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        font-size: 0.82rem;
        color: var(--text-2) !important;
        line-height: 1.55;
        margin-top: 0.5rem;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: var(--text-2) !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] {
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.20), 0 0 12px rgba(34, 211, 238, 0.5) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="input"] > div,
    [data-testid="stSidebar"] input[type="number"] {
        background: var(--surface-2) !important;
        border: 1px solid var(--border-2) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="input"] > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.16) !important;
    }

    /* ═══ HERO ═══ */
    .hero {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        background: linear-gradient(135deg, rgba(16, 20, 31, 0.85) 0%, rgba(22, 27, 41, 0.85) 100%);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 1.9rem 2.25rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }
    .hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #22d3ee, #10b981, #8b5cf6, #f472b6);
    }
    .hero::after {
        content: '';
        position: absolute;
        top: -50%; right: -10%;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(34, 211, 238, 0.10), transparent 65%);
        pointer-events: none;
    }
    .hero-icon {
        flex-shrink: 0;
        width: 72px; height: 72px;
        border-radius: 20px;
        background: linear-gradient(135deg, #22d3ee 0%, #10b981 55%, #8b5cf6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        box-shadow: 0 10px 32px rgba(34, 211, 238, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.25);
        position: relative;
        z-index: 1;
    }
    .hero-label {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent);
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero-label::before {
        content: '';
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 10px var(--accent);
        animation: pulse-dot 1.8s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%      { opacity: 0.5; transform: scale(0.85); }
    }
    .hero-text { position: relative; z-index: 1; }
    .hero-text h1 {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: var(--text) !important;
        margin: 0 0 0.5rem 0 !important;
        line-height: 1.05 !important;
        letter-spacing: -0.03em !important;
        background: linear-gradient(90deg, #ffffff 0%, #b8e6f0 55%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-text p {
        color: var(--muted) !important;
        font-size: 1rem !important;
        margin: 0 !important;
        line-height: 1.6 !important;
        max-width: 620px;
    }

    /* ═══ CAMERA INPUT ═══ */
    [data-testid="stCameraInput"] {
        background: linear-gradient(135deg, rgba(16, 20, 31, 0.7), rgba(22, 27, 41, 0.7)) !important;
        border: 1px solid var(--border) !important;
        border-radius: 20px !important;
        padding: 1.25rem !important;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }
    [data-testid="stCameraInput"] button {
        background: linear-gradient(135deg, #22d3ee, #10b981) !important;
        color: #041a1a !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
        box-shadow: 0 4px 18px rgba(34, 211, 238, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stCameraInput"] button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 26px rgba(34, 211, 238, 0.55) !important;
    }
    [data-testid="stCameraInput"] video {
        border-radius: 14px !important;
        border: 1px solid var(--border-2) !important;
    }

    /* ═══ CONTENEDOR CON BORDE ═══ */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, rgba(16, 20, 31, 0.75) 0%, rgba(22, 27, 41, 0.75) 100%) !important;
        border: 1px solid var(--border) !important;
        border-radius: 20px !important;
        padding: 1.35rem 1.5rem !important;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }

    .card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.02rem;
        font-weight: 700;
        color: var(--text);
        margin: 0 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .card-title .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.15), 0 0 12px var(--accent);
    }

    /* ═══ MÉTRICAS ═══ */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat {
        background: linear-gradient(135deg, rgba(16, 20, 31, 0.85) 0%, rgba(22, 27, 41, 0.85) 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.03);
        transition: all 0.2s ease;
    }
    .stat::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #22d3ee, #10b981, #8b5cf6);
    }
    .stat:hover {
        border-color: rgba(34, 211, 238, 0.35);
        box-shadow: 0 8px 40px rgba(34, 211, 238, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    .stat-label {
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted) !important;
        font-weight: 700;
        margin-bottom: 0.65rem;
    }
    .stat-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1;
        letter-spacing: -0.03em;
        text-shadow: 0 0 24px rgba(34, 211, 238, 0.18);
    }
    .stat-value small {
        font-size: 1rem;
        color: var(--accent);
        font-weight: 500;
        margin-left: 0.15rem;
    }

    /* ═══ LISTA DE DETECCIONES ═══ */
    .det-list { display: flex; flex-direction: column; gap: 0.55rem; }
    .det-item {
        display: grid;
        grid-template-columns: 1fr auto auto;
        align-items: center;
        gap: 0.9rem;
        padding: 0.8rem 1rem;
        background: rgba(28, 34, 51, 0.55);
        border: 1px solid var(--border);
        border-radius: 12px;
        transition: all 0.18s ease;
    }
    .det-item:hover {
        background: rgba(34, 211, 238, 0.06);
        border-color: rgba(34, 211, 238, 0.35);
        transform: translateX(2px);
        box-shadow: 0 4px 20px rgba(34, 211, 238, 0.10);
    }
    .det-name {
        font-weight: 600;
        color: var(--text);
        font-size: 0.94rem;
        text-transform: capitalize;
    }
    .det-count {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.25rem 0.7rem;
        border-radius: 100px;
        background: rgba(34, 211, 238, 0.10);
        color: var(--accent) !important;
        border: 1px solid rgba(34, 211, 238, 0.22);
        white-space: nowrap;
    }
    .det-conf {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        min-width: 90px;
        justify-content: flex-end;
    }
    .det-conf-bar {
        width: 46px; height: 6px;
        background: rgba(148, 163, 184, 0.14);
        border-radius: 100px;
        overflow: hidden;
    }
    .det-conf-fill {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #10b981, #22d3ee);
        box-shadow: 0 0 10px rgba(34, 211, 238, 0.5);
    }
    .det-conf-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-2) !important;
        min-width: 36px;
        text-align: right;
    }

    /* ═══ BOTONES ═══ */
    .stButton > button {
        background: linear-gradient(135deg, #22d3ee, #10b981) !important;
        color: #041a1a !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 4px 18px rgba(34, 211, 238, 0.30) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button p { color: #041a1a !important; }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 28px rgba(34, 211, 238, 0.55) !important;
    }

    /* ═══ IMÁGENES ═══ */
    [data-testid="stImage"] img {
        border-radius: 14px !important;
        border: 1px solid var(--border-2) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    /* ═══ DATAFRAME ═══ */
    [data-testid="stDataFrame"] {
        border-radius: 14px !important;
        border: 1px solid var(--border-2) !important;
        overflow: hidden;
        background: var(--surface-2) !important;
    }

    /* ═══ ALERTAS ═══ */
    [data-testid="stAlert"] {
        border-radius: 14px !important;
        border: 1px solid var(--border-2) !important;
        background: rgba(22, 27, 41, 0.85) !important;
        color: var(--text) !important;
    }
    [data-testid="stAlert"] svg { fill: var(--accent) !important; }

    /* ═══ EXPANDER ═══ */
    [data-testid="stExpander"] {
        background: rgba(22, 27, 41, 0.7) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        overflow: hidden;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }
    [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: var(--text) !important;
        padding: 0.9rem 1.1rem !important;
        transition: color 0.15s ease;
    }
    [data-testid="stExpander"] summary:hover { color: var(--accent) !important; }
    [data-testid="stExpander"] summary svg { fill: var(--accent) !important; }

    /* ═══ SPINNER ═══ */
    .stSpinner > div {
        border-top-color: var(--accent) !important;
    }

    /* ═══ SEPARADOR ═══ */
    hr {
        border-color: var(--border) !important;
        opacity: 0.6;
    }

    /* ═══ CAPTION ═══ */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
        font-size: 0.82rem !important;
    }

    /* ═══ SCROLLBAR ═══ */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #22d3ee, #10b981);
        border-radius: 10px;
        border: 2px solid var(--bg);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #8b5cf6, #22d3ee);
    }

    @media (max-width: 720px) {
        .hero { flex-direction: column; text-align: center; padding: 1.5rem; }
        .stats-grid { grid-template-columns: 1fr; }
        .hero-text h1 { font-size: 1.7rem !important; }
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-icon">🔍</div>
    <div class="hero-text">
        <div class="hero-label">Visión por computadora · YOLOv5</div>
        <h1>Detección de Objetos</h1>
        <p>Captura una imagen con tu cámara y detecta los objetos que aparecen en ella en segundos. Ajusta los parámetros desde la barra lateral.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CARGA DEL MODELO
# ═══════════════════════════════════════════════════════════════
with st.spinner("Cargando modelo YOLOv5..."):
    model = load_model()

if not model:
    st.error("No se pudo cargar el modelo. Verifica las dependencias e inténtalo nuevamente.")
    st.stop()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-title">⚙️ Parámetros</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Configuración de detección</div>', unsafe_allow_html=True)

    conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01,
                                help="Filtra las detecciones con menos seguridad que este umbral.")
    iou_threshold  = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01,
                                help="Controla el solapamiento permitido entre cajas.")
    max_det        = st.number_input("Detecciones máximas", 10, 2000, 1000, 10)

    st.markdown('---')
    st.markdown('<div class="sb-hint">💡 Si no detecta nada, prueba bajar la confianza mínima a 0.15 o menos.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CÁMARA
# ═══════════════════════════════════════════════════════════════
picture = st.camera_input("Capturar imagen", key="camera")


# ═══════════════════════════════════════════════════════════════
# RESULTADOS
# ═══════════════════════════════════════════════════════════════
if picture:
    bytes_data = picture.getvalue()
    pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    np_img  = np.array(pil_img)[..., ::-1]  # RGB → BGR

    with st.spinner("Detectando objetos..."):
        try:
            results = model(np_img, conf=conf_threshold, iou=iou_threshold, max_det=int(max_det))
        except Exception as e:
            st.error(f"Error durante la detección: {str(e)}")
            st.stop()

    result = results[0]
    boxes  = result.boxes
    annotated = result.plot()
    annotated_rgb = annotated[:, :, ::-1]  # BGR → RGB

    if boxes is not None and len(boxes) > 0:
        label_names = model.names
        category_count = {}
        category_conf  = {}

        for box in boxes:
            cat  = int(box.cls.item())
            conf = float(box.conf.item())
            category_count[cat] = category_count.get(cat, 0) + 1
            category_conf.setdefault(cat, []).append(conf)

        total_obj = sum(category_count.values())
        total_cls = len(category_count)
        avg_conf  = float(np.mean([np.mean(c) for c in category_conf.values()]))

        items = sorted(category_count.items(), key=lambda x: -x[1])

        # ── Estadísticas ──
        st.markdown(f"""
            <div class="stats-grid">
                <div class="stat">
                    <div class="stat-label">Objetos detectados</div>
                    <div class="stat-value">{total_obj}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Clases distintas</div>
                    <div class="stat-value">{total_cls}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Confianza promedio</div>
                    <div class="stat-value">{int(avg_conf*100)}<small>%</small></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1.15, 1], gap="medium")

        with col1:
            with st.container(border=True):
                st.markdown(
                    '<div class="card-title"><span class="dot"></span>Imagen con detecciones</div>',
                    unsafe_allow_html=True
                )
                st.image(annotated_rgb, use_container_width=True)

        with col2:
            with st.container(border=True):
                st.markdown(
                    '<div class="card-title"><span class="dot"></span>Objetos detectados</div>',
                    unsafe_allow_html=True
                )
                items_html = '<div class="det-list">'
                for cat, count in items:
                    avg_c = float(np.mean(category_conf[cat]))
                    pct = int(avg_c * 100)
                    items_html += f"""
                        <div class="det-item">
                            <div class="det-name">{label_names[cat]}</div>
                            <div class="det-count">× {count}</div>
                            <div class="det-conf">
                                <div class="det-conf-bar"><div class="det-conf-fill" style="width:{pct}%"></div></div>
                                <div class="det-conf-val">{pct}%</div>
                            </div>
                        </div>
                    """
                items_html += '</div>'
                st.markdown(items_html, unsafe_allow_html=True)

        with st.expander("📊 Ver tabla de datos y gráfico"):
            data = [
                {
                    "Categoría": label_names[cat],
                    "Cantidad": count,
                    "Confianza promedio": f"{np.mean(category_conf[cat]):.2f}"
                }
                for cat, count in items
            ]
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df.set_index("Categoría")["Cantidad"])

    else:
        st.markdown("""
            <div class="stat" style="text-align:center; padding:2.5rem 1.5rem;">
                <div style="font-size:3rem; margin-bottom:0.75rem;">🤔</div>
                <div style="font-family:'Outfit',sans-serif; font-weight:700; font-size:1.15rem; color:#e8ecf6; margin-bottom:0.5rem;">
                    No se detectaron objetos
                </div>
                <div style="color:#7d879c; font-size:0.94rem;">
                    Prueba a reducir el umbral de confianza en la barra lateral.
                </div>
            </div>
        """, unsafe_allow_html=True)


st.markdown("---")
st.caption("**Acerca de la aplicación**: Detección de objetos con YOLOv5 + Streamlit + PyTorch.")
