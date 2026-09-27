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
# ESTILOS — SOFT AURORA STUDIO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {
        --surface: #ffffff;
        --border: #eceef5;
        --border-2: #e2e5f0;
        --text: #14172a;
        --muted: #6e7391;
        --muted-2: #9297b3;
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --primary-soft: #eef0ff;
        --pink: #ec4899;
        --amber: #f59e0b;
    }

    html, body, [class*="css"], .stApp, button, input, textarea, select {
        font-family: 'Inter', sans-serif !important;
    }
    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em !important;
    }

    /* ═══ FONDO AURORA ═══ */
    .stApp {
        background:
            radial-gradient(ellipse at top left,  rgba(99, 102, 241, 0.10), transparent 45%),
            radial-gradient(ellipse at top right, rgba(236, 72, 153, 0.07), transparent 45%),
            radial-gradient(ellipse at bottom,    rgba(245, 158, 11, 0.05), transparent 55%),
            #f6f7fb !important;
        background-attachment: fixed;
    }

    /* ═══ SIDEBAR ═══ */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    .sb-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text);
        margin: 0.25rem 0 1.25rem 0;
    }
    .sb-section {
        font-size: 0.7rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 700;
        color: var(--muted-2);
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
    }
    .sb-hint {
        background: var(--primary-soft);
        border-left: 3px solid var(--primary);
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        font-size: 0.82rem;
        color: #4c4f76 !important;
        line-height: 1.5;
        margin-top: 0.5rem;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #4c4f76 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] {
        background-color: var(--primary) !important;
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.18) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="slider"] > div > div > div {
        background: var(--primary) !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="input"] > div,
    [data-testid="stSidebar"] input[type="number"] {
        background: #ffffff !important;
        border: 1px solid var(--border-2) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="input"] > div:focus-within {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.14) !important;
    }

    /* ═══ HERO ═══ */
    .hero {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 1.9rem 2.25rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 24px rgba(20, 23, 42, 0.05);
    }
    .hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899, #f59e0b);
    }
    .hero-icon {
        flex-shrink: 0;
        width: 72px; height: 72px;
        border-radius: 20px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #ec4899 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        box-shadow: 0 10px 28px rgba(99, 102, 241, 0.32);
    }
    .hero-label {
        display: inline-block;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--primary);
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .hero-text h1 {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: var(--text) !important;
        margin: 0 0 0.5rem 0 !important;
        line-height: 1.05 !important;
        letter-spacing: -0.03em !important;
    }
    .hero-text p {
        color: var(--muted) !important;
        font-size: 1rem !important;
        margin: 0 !important;
        line-height: 1.55 !important;
        max-width: 620px;
    }

    /* ═══ CAMERA INPUT ═══ */
    [data-testid="stCameraInput"] {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 20px !important;
        padding: 1.25rem !important;
        box-shadow: 0 4px 24px rgba(20, 23, 42, 0.05);
    }
    [data-testid="stCameraInput"] button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stCameraInput"] button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 22px rgba(99, 102, 241, 0.45) !important;
    }
    [data-testid="stCameraInput"] video {
        border-radius: 14px !important;
    }

    /* ═══ CONTENEDOR CON BORDE (st.container(border=True)) ═══ */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 20px !important;
        padding: 1.35rem 1.5rem !important;
        box-shadow: 0 4px 24px rgba(20, 23, 42, 0.05) !important;
    }

    .card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.02rem;
        font-weight: 700;
        color: var(--text);
        margin: 0 0 0.9rem 0;
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }
    .card-title .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--primary);
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.14);
    }

    /* ═══ MÉTRICAS ═══ */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.25rem 1.35rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(20, 23, 42, 0.05);
    }
    .stat::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
    }
    .stat-label {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted-2);
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .stat-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1;
        letter-spacing: -0.03em;
    }
    .stat-value small {
        font-size: 0.95rem;
        color: var(--muted);
        font-weight: 500;
        margin-left: 0.15rem;
    }

    /* ═══ LISTA DE DETECCIONES ═══ */
    .det-list { display: flex; flex-direction: column; gap: 0.6rem; }
    .det-item {
        display: grid;
        grid-template-columns: 1fr auto auto;
        align-items: center;
        gap: 0.9rem;
        padding: 0.8rem 1rem;
        background: linear-gradient(135deg, #fafbff 0%, #ffffff 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        transition: all 0.18s ease;
    }
    .det-item:hover {
        border-color: #c7ccf5;
        transform: translateX(2px);
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.10);
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
        padding: 0.25rem 0.65rem;
        border-radius: 100px;
        background: var(--primary-soft);
        color: var(--primary-dark);
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
        width: 44px; height: 6px;
        background: #eceef5;
        border-radius: 100px;
        overflow: hidden;
    }
    .det-conf-fill {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }
    .det-conf-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--muted);
        min-width: 36px;
        text-align: right;
    }

    /* ═══ BOTONES ═══ */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.28) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button p { color: #ffffff !important; }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.42) !important;
    }

    /* ═══ IMÁGENES ═══ */
    [data-testid="stImage"] img {
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
    }

    /* ═══ DATAFRAME ═══ */
    [data-testid="stDataFrame"] {
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
        overflow: hidden;
    }

    /* ═══ ALERTAS ═══ */
    [data-testid="stAlert"] {
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
        background: #ffffff !important;
        color: var(--text) !important;
    }

    /* ═══ EXPANDER ═══ */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(20, 23, 42, 0.04);
    }
    [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: var(--text) !important;
        padding: 0.9rem 1.1rem !important;
    }

    /* ═══ SCROLLBAR ═══ */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #f6f7fb; }
    ::-webkit-scrollbar-thumb {
        background: #d7dbeb;
        border-radius: 10px;
        border: 2px solid #f6f7fb;
    }
    ::-webkit-scrollbar-thumb:hover { background: #a5aac9; }

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

        # Ordenar por cantidad descendente
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

        # ── Tabla + gráfico en un expander (por compatibilidad) ──
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
                <div style="font-family:'Outfit',sans-serif; font-weight:700; font-size:1.15rem; color:#14172a; margin-bottom:0.5rem;">
                    No se detectaron objetos
                </div>
                <div style="color:#6e7391; font-size:0.94rem;">
                    Prueba a reducir el umbral de confianza en la barra lateral.
                </div>
            </div>
        """, unsafe_allow_html=True)


st.markdown("---")
st.caption("**Acerca de la aplicación**: Detección de objetos con YOLOv5 + Streamlit + PyTorch.")
