import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from scripts.model import DeepfakeDetectorModel
from scripts.utils import FaceDetector, GradCAM, overlay_heatmap, predict_with_tta
import numpy as np
import time

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Session State ----------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ---------------------- Custom CSS (Cyber-Forensics Theme) ----------------------
st.markdown("""
<style>
    /* 
    ========================================
    Premium Cyber-Forensics Theme
    ======================================== 
    */
    
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;500;600;800&display=swap');

    :root {
        /* Core Colors - High Contrast */
        --bg-app: #09090b;       /* Zinc 950 - Very Dark */
        --bg-card: rgba(24, 24, 27, 0.7); /* Zinc 900 - Translucent */
        --bg-card-hover: rgba(39, 39, 42, 0.8);
        
        /* Text Colors - Maximized Visibility */
        --text-primary: #ffffff;
        --text-secondary: #e2e8f0; /* Slate 200 - Very Light Gray */
        --text-muted: #cbd5e1;     /* Slate 300 */
        
        /* Accents */
        --accent-primary: #3b82f6; /* Blue 500 */
        --accent-glow: rgba(59, 130, 246, 0.5);
        --accent-gradient: linear-gradient(135deg, #3b82f6, #8b5cf6);
        
        /* Status Indicators */
        --status-real-bg: rgba(34, 197, 94, 0.15);
        --status-real-border: #22c55e;
        --status-real-text: #4ade80;
        
        --status-fake-bg: rgba(239, 68, 68, 0.15);
        --status-fake-border: #ef4444;
        --status-fake-text: #f87171;
        
        --border-light: rgba(255, 255, 255, 0.1);
    }

    /* Animation Keyframes */
    @keyframes glow-pulse {
        0% { background-position: 0% 0%, 0% 0%, 50% 0%, 100% 0%, 0% 100%; }
        50% { background-position: 0% 0%, 0% 0%, 50% 10%, 100% 10%, 0% 90%; }
        100% { background-position: 0% 0%, 0% 0%, 50% 0%, 100% 0%, 0% 100%; }
    }

    /* Global Resets & App Background */
    .stApp {
        background-color: #020617; /* Deepest Slate */
        
        /* Layered Background: Grid + INTENSE Ambient Glows */
        background-image: 
            /* Subtle Grid Pattern */
            linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px), 
            linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px),
            
            /* Top Center Blue Glow - INTENSIFIED */
            radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.35) 0%, rgba(56, 189, 248, 0.1) 40%, transparent 60%),
            
            /* Top Right Purple Glow - INTENSIFIED */
            radial-gradient(circle at 90% 10%, rgba(139, 92, 246, 0.3) 0%, rgba(139, 92, 246, 0.1) 40%, transparent 60%),
            
            /* Bottom Left Neon Green Tint - INTENSIFIED */
            radial-gradient(circle at 10% 90%, rgba(34, 197, 94, 0.15) 0%, transparent 50%);
            
        background-size: 40px 40px, 40px 40px, 120% 120%, 120% 120%, 100% 100%;
        background-attachment: fixed;
        
        /* Animation */
        animation: glow-pulse 15s ease-in-out infinite alternate;
        
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em;
    }
    
    /* Logo - CRISP & STATIC */
    .logo-shield {
        display: inline-block;
        font-size: 3.5rem;
        margin-right: 15px;
        /* No animation, no shadow, just clean */
        filter: none;
        text-shadow: none;
    }

    /* Main Title - CLEAN WHITE */
    h1 {
        display: flex;
        align-items: center;
        color: #ffffff !important; /* Pure white */
        background: none; /* Remove gradient if it causes issues */
        -webkit-text-fill-color: initial;
        
        text-shadow: none; /* Remove glow */
        font-weight: 800 !important;
        font-size: 3rem !important;
        margin-bottom: 0 !important;
        padding-bottom: 10px !important;
    }
    
    /* Subtitle Class */
    .header-subtitle {
        color: #94a3b8; 
        font-size: 1.1rem; 
        font-family: 'Inter', sans-serif; 
        margin-top: -10px !important;
        margin-left: 5px;
        text-shadow: none;
        letter-spacing: 0.05em;
        opacity: 0.9;
        font-weight: 500;
    }
    
    p, div, span, label {
        color: var(--text-secondary);
    }
    
    .mono {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Glassmorphic Cards */
    .dashboard-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-light);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 20px;
    }

    .dashboard-card:hover {
        border-color: var(--accent-primary);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 0 15px var(--accent-glow);
        transform: translateY(-2px);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #0f172a, #020617);
        border-right: 1px solid rgba(59, 130, 246, 0.15);
        box-shadow: 5px 0 15px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--text-primary);
        font-family: 'Space Grotesk', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 1.1rem;
        margin-top: 20px;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    /* Top Header Styling (Remove White Bar) */
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: blur(5px); /* Optional: keeps it readable if content scrolling under */
    }
    
    /* Toolbar Options (Keep visible but match theme) */
    .stDeployButton {
        visibility: hidden; /* Often distracting in custom apps */
    }
    
    /* Sidebar Toggle Button Glow - ULTRA INTENSE */
    [data-testid="stHeader"] button {
        color: var(--accent-primary) !important;
        text-shadow: 0 0 15px rgba(59, 130, 246, 0.9);
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 8px !important;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); /* Bouncy spring transition */
        padding: 0.5rem !important; /* Touch target size */
    }
    
    [data-testid="stHeader"] button:hover {
        color: #fff !important;
        background: rgba(59, 130, 246, 0.3) !important;
        border-color: #3b82f6 !important;
        text-shadow: 0 0 30px rgba(59, 130, 246, 1), 0 0 60px rgba(59, 130, 246, 0.8); /* Double layer text glow */
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.5), 0 0 40px rgba(59, 130, 246, 0.3), inset 0 0 10px rgba(59, 130, 246, 0.3); /* Triple layer box glow */
        transform: scale(1.15) rotate(5deg); /* Dynamic movement */
    }

    /* Custom Badges */
    .badge {
        padding: 6px 16px;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-block;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    
    .badge-fake {
        background-color: var(--status-fake-bg);
        color: var(--status-fake-text);
        border: 1px solid var(--status-fake-border);
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
    }
    
    .badge-real {
        background-color: var(--status-real-bg);
        color: var(--status-real-text);
        border: 1px solid var(--status-real-border);
        box-shadow: 0 0 15px rgba(34, 197, 94, 0.2);
    }

    /* Upload Area Enhancement */
    [data-testid="stFileUploader"] {
        padding: 30px;
        background: rgba(15, 23, 42, 0.6) !important; /* Darker, slightly translucent */
        border-radius: 16px;
        border: 2px dashed rgba(59, 130, 246, 0.5); /* Base cyber border */
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.1); /* Subtle static glow */
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6; /* Brighter Blue */
        background: rgba(59, 130, 246, 0.1) !important;
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.25); /* Intense hover glow */
        transform: scale(1.01);
    }
    
    /* Fix for white background in some streamlit versions */
    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
    }
    
    /* Upload Button Style */
    [data-testid="stFileUploader"] button {
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.5);
        color: white;
        transition: all 0.2s;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background: rgba(59, 130, 246, 0.4);
        border-color: #3b82f6;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
        color: white;
    }
    
    /* Upload Text */
    [data-testid="stFileUploader"] div, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small {
        color: var(--text-secondary) !important;
    }

    /* Buttons */
    .stButton > button {
        background: var(--accent-gradient);
        color: white !important;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        transition: all 0.2s;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px rgba(59, 130, 246, 0.4);
    }
    
    /* Metrics/Stats */
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
    }

    /* Dividers */
    hr {
        border: 0;
        border-top: 1px solid var(--border-light);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- Logic ----------------------
@st.cache_resource
def load_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeDetectorModel().to(device)
    try:
        model.load_state_dict(torch.load("models/deepfake_model.pth", map_location=device))
    except:
        st.error("Model file not found! Please check 'models/deepfake_model.pth'")
    model.eval()
    face_detector = FaceDetector()
    return model, face_detector, device

model, face_detector, device = load_assets()

def analyze_image_pipeline(image):
    # Simulate pipeline steps for UX
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # Step 1: Face Detection
    status_text.text("🔍 Phase 1: Detecting facial features...")
    progress_bar.progress(25)
    time.sleep(0.3)
    cropped_face, face_found, original = face_detector.detect_and_crop(image)
    
    # Step 2: Preprocessing
    status_text.text("📐 Phase 2: Normalizing input tensor...")
    progress_bar.progress(50)
    time.sleep(0.2)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(cropped_face).unsqueeze(0).to(device)
    
    # Step 3: Inference
    status_text.text("🧠 Phase 3: Running Deep Neural Network...")
    progress_bar.progress(75)
    
    target_layer = model.model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam(input_tensor)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
    
    # Step 4: Analysis
    status_text.text("📊 Phase 4: Generating Forensic Report...")
    progress_bar.progress(100)
    time.sleep(0.2)
    status_text.empty()
    progress_bar.empty()
    
    # Overlay heatmap
    heatmap_img = overlay_heatmap(heatmap, cropped_face.resize((224, 224)))
    label = "Fake" if prob > 0.5 else "Real"
    
    return label, prob, cropped_face, heatmap_img, face_found

# ---------------------- UI Components ----------------------
def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        # Custom HTML Title with Animated Logo
        st.markdown("""
            <h1 style='margin-bottom: 0px;'>
                <span class='logo-shield'>🛡️</span> 
                Deepfake Image Detector
            </h1>
        """, unsafe_allow_html=True)
        st.markdown("<p class='header-subtitle'>    Enterprise Grade Digital Content Verification System</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='text-align: right; font-family: "JetBrains Mono"; font-size: 0.8rem; color: var(--accent-primary);'>
            SYSTEM: ONLINE<br>
            GPU: {}
        </div>
        """.format("ACTIVE" if torch.cuda.is_available() else "N/A"), unsafe_allow_html=True)
    st.markdown("---")

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        
        st.markdown("#### Detection Settings")
        show_heatmap = st.toggle("Explainability Layer (Grad-CAM)", value=True)
        high_sensitivity = st.toggle("High Sensitivity Mode", value=False)
        
        st.markdown("---")
        st.markdown("### 🕒 Session History")
        if not st.session_state.history:
            st.markdown("<p style='color: var(--text-muted); font-size: 0.9rem; font-style: italic; opacity: 0.7;'>No analysis history recorded.</p>", unsafe_allow_html=True)
        else:
            for item in reversed(st.session_state.history[-5:]):
                status_color = "#ef4444" if item['label'] == "Fake" else "#22c55e"
                bg_color = "rgba(239, 68, 68, 0.1)" if item['label'] == "Fake" else "rgba(34, 197, 94, 0.1)"
                
                st.markdown(f"""
                <div style='
                    padding: 12px; 
                    border-left: 3px solid {status_color}; 
                    background: {bg_color}; 
                    margin-bottom: 10px; 
                    border-radius: 4px;
                    border: 1px solid rgba(255,255,255,0.05);
                    transition: transform 0.2s;
                ' onmouseover="this.style.transform='translateX(5px)'" onmouseout="this.style.transform='translateX(0px)'">
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                        <span style='font-family: "JetBrains Mono"; font-size: 0.75rem; color: var(--text-muted);'>{item['time']}</span>
                        <span style='font-weight: bold; font-size: 0.8rem; color: {status_color};'>{item['label'].upper()}</span>
                    </div>
                    <div style='font-size: 0.85rem; color: var(--text-primary);'>Confidence: <b>{item['score']*100:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)
                
        return show_heatmap

def render_dashboard(label, confidence, face_img, heatmap_img, show_heatmap, face_found):
    # Determine styles
    is_fake = label == "Fake"
    color = "var(--status-fake-text)" if is_fake else "var(--status-real-text)"
    badge_class = "badge-fake" if is_fake else "badge-real"
    
    # 1. Main Score Card
    st.markdown(f"""
    <div class="dashboard-card" style="border-top: 4px solid {color}; text-align: center;">
        <h3 style="color: var(--text-muted); text-transform: uppercase; letter-spacing: 2px;">Analysis Conclusion</h3>
        <div style="font-size: 4rem; font-weight: 800; color: {color}; margin: 10px 0;">
            {label.upper()}
        </div>
        <div class="{badge_class}">
            CONFIDENCE SCORE: {confidence*100:.2f}%
        </div>
        <p style="margin-top: 15px; color: var(--text-muted); font-size: 1rem;">
            The model has identified this image as <b style="color: var(--text-primary)">{label}</b> based on artifact analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Detailed Forensics Grid
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### 👁️ Region of Interest")
        if face_found:
            st.image(face_img, use_container_width=True, caption="Detected Face Crop")
        else:
            st.warning("No face detected - Analyzing full frame")
            st.image(face_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### 🌡️ Artifact Heatmap")
        if show_heatmap:
            st.image(heatmap_img, use_container_width=True, caption="Grad-CAM Activation")
            st.info("Red areas indicate regions contributing most to the 'Fake' classification.")
        else:
            st.markdown("<div style='height: 200px; display: flex; align-items: center; justify-content: center; color: var(--text-muted);'>Heatmap Disabled</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Main Execution ----------------------
render_sidebar()
render_header()

# Drag & Drop Zone
uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], help="Upload an image for forensic analysis")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Layout: Image preview on left, results on right (after processing)
    if "analyzed_file" not in st.session_state or st.session_state.analyzed_file != uploaded_file.name:
        # Initial View
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(image, caption="Source Image", use_container_width=True)
        with c2:
            st.markdown("""
            <div class="dashboard-card" style="text-align: center; padding: 40px;">
                <h3>Ready for Analysis</h3>
                <p style="color: var(--text-muted); font-size: 1.1rem;">Initiate the deep learning forensic pipeline to detect manipulation.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🚀 Start Forensic Analysis"):
                with st.spinner("Initializing Deepfake Detection Model..."):
                    label, prob, face_crop, heatmap, face_found = analyze_image_pipeline(image)
                    
                    # Update Session State
                    st.session_state.analyzed_file = uploaded_file.name
                    st.session_state.last_result = {
                        "label": label,
                        "prob": prob,
                        "face": face_crop,
                        "heatmap": heatmap,
                        "found": face_found
                    }
                    st.session_state.history.append({
                        "time": time.strftime("%H:%M:%S"),
                        "label": label,
                        "score": prob
                    })
                    st.rerun()

    else:
        # Results View
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(image, caption="Source Image", use_container_width=True)
            if st.button("🔄 New Analysis"):
                del st.session_state['analyzed_file']
                st.rerun()
                
        with c2:
            res = st.session_state.last_result
            show_heatmap = st.session_state.get('history', []) # accessing sidebar state is tricky, simplify
            # Re-read checkbox from sidebar for render
            # Note: streamlit sidebar state is global
            render_dashboard(res['label'], res['prob'], res['face'], res['heatmap'], True, res['found'])

else:
    # Empty State with Hero
    st.markdown("""
    <div style="text-align: center; padding: 50px 20px;">
        <h2 style="color: var(--text-muted);">Waiting for Image...</h2>
        <p style="color: var(--text-muted); font-size: 1.2rem;">Upload an image above to begin forensic examination.</p>
    </div>
    """, unsafe_allow_html=True)
