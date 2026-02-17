# ==========================================================
# DermalScan v4.0 — Advanced Webcam with Snapshot & Analysis
# ==========================================================
import streamlit as st
import numpy as np
import cv2
import pandas as pd
import time
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from random import randint
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from datetime import datetime

# ----------------------------------------------------------
# 🌐 PAGE CONFIGURATION (MUST BE FIRST!)
# ----------------------------------------------------------
st.set_page_config(page_title="DermalScan AI", layout="wide")

# 🎨 Crimson Quantum Theme
st.markdown("""
<style>
.stApp {background-color: #4B0E1E; color: #F8F8FF;}
h1, h2, h3, h4, h5, h6 {color: #ffffff; text-shadow: 0px 0px 8px rgba(255,255,255,0.4);}
.stFileUploader {background-color: #5B1B3B; border: 1px solid #9A4D7F; border-radius: 12px; padding: 1rem;}
.stButton > button {
    background: linear-gradient(90deg,#7B2CBF,#9D4EDD);
    color: #FFFFFF !important;
    border-radius: 8px; border: none;
    font-weight: 600;
    padding: 0.6em 1.4em;
    box-shadow: 0 0 10px rgba(157,78,221,0.6);
}
.stButton > button:hover {
    background: linear-gradient(90deg,#9D4EDD,#7B2CBF);
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(157,78,221,0.8);
}
.css-1r6slb0 {background-color: #5B1B3B !important; border-radius: 12px !important; border: 1px solid #9A4D7F !important;}
footer {visibility: hidden;}
.stDownloadButton button {
    color: #4B0E1E !important;
    background-color: #F8F8FF !important;
    border-radius: 8px !important;
    font-weight: 600;
}
.stDownloadButton button:hover {
    background-color: #EFBBD2 !important;
}
.recommendation-box {
    background: rgba(155,55,155,0.15);
    border: 1px solid rgba(157,78,221,0.4);
    border-radius: 15px;
    padding: 1.5em;
    margin: 1em 0;
    box-shadow: 0 0 20px rgba(157,78,221,0.2);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# ⚡ CACHE HEAVY COMPONENTS
# ----------------------------------------------------------
@st.cache_resource
def load_ai_components():
    model = load_model("densenet121_best_optimized.h5")
    detector = MTCNN()
    classes = ["clear face", "darkspots", "puffy eyes", "wrinkles"]
    return model, detector, classes

model, detector, classes = load_ai_components()

# ----------------------------------------------------------
# 💡 SKINCARE RECOMMENDATION ENGINE
# ----------------------------------------------------------
def get_skincare_recommendations(feature, confidence, age):
    """Generate personalized skincare recommendations based on analysis"""
    
    recommendations = {
        "clear face": {
            "routine": [
                "Morning: Gentle cleanser → Vitamin C serum → Moisturizer → SPF 50+",
                "Evening: Cleanser → Hyaluronic acid serum → Night cream"
            ],
            "products": [
                "Gentle foaming cleanser (pH balanced)",
                "Vitamin C serum for brightness",
                "Lightweight moisturizer with SPF",
                "Hydrating night cream"
            ],
            "sleep": "7-9 hours per night for optimal skin repair",
            "water": "2.5-3 liters daily for skin hydration",
            "tips": [
                "Maintain your current routine consistently",
                "Protect from sun damage with daily SPF",
                "Consider monthly professional facials",
                "Use silk pillowcases to prevent sleep lines"
            ]
        },
        "darkspots": {
            "routine": [
                "Morning: Cleanser → Vitamin C serum → Niacinamide → SPF 50+",
                "Evening: Cleanser → Alpha Arbutin serum → Retinol (2-3x/week) → Night cream"
            ],
            "products": [
                "Brightening cleanser with glycolic acid",
                "Vitamin C + Niacinamide serum (10%)",
                "Alpha Arbutin or Kojic acid treatment",
                "Retinol 0.5% (build tolerance gradually)",
                "High SPF sunscreen (reapply every 2 hours)"
            ],
            "sleep": "8-9 hours per night - critical for skin cell turnover",
            "water": "3-3.5 liters daily to flush toxins and reduce pigmentation",
            "tips": [
                "NEVER skip sunscreen - UV exposure worsens dark spots",
                "Use a wide-brimmed hat outdoors",
                "Consider professional treatments: Chemical peels, Laser therapy",
                "Be patient - visible results take 8-12 weeks",
                "Avoid picking at spots to prevent scarring"
            ]
        },
        "puffy eyes": {
            "routine": [
                "Morning: Cold compress (5 min) → Eye cream with caffeine → SPF",
                "Evening: Gentle eye makeup remover → Peptide eye cream → Sleep on elevated pillow"
            ],
            "products": [
                "Caffeine-based eye serum or cream",
                "Cooling gel eye masks (refrigerate)",
                "Vitamin K eye cream for dark circles",
                "Jade roller or cooling eye massager",
                "Hyaluronic acid for hydration"
            ],
            "sleep": "7-8 hours with head slightly elevated (add extra pillow)",
            "water": "2-2.5 liters daily (avoid excess at night to reduce morning puffiness)",
            "tips": [
                "Reduce salt intake to minimize fluid retention",
                "Sleep on your back to prevent fluid pooling",
                "Use cold spoons or chilled cucumber slices (10 min)",
                "Limit alcohol and caffeine before bed",
                "Gently massage eye area to improve lymphatic drainage",
                "Manage allergies - they worsen puffiness"
            ]
        },
        "wrinkles": {
            "routine": [
                "Morning: Cleanser → Vitamin C → Peptide serum → Hyaluronic acid → SPF 50+",
                "Evening: Cleanser → Retinol (every night) → Peptide cream → Rich night moisturizer"
            ],
            "products": [
                "Retinol 1% or Tretinoin (prescription strength)",
                "Peptide complex serum (Matrixyl, Argireline)",
                "Hyaluronic acid serum for plumping",
                "Rich moisturizer with ceramides",
                "Eye cream with retinol + peptides",
                "Sunscreen SPF 50+ (zinc oxide based)"
            ],
            "sleep": "8-9 hours - sleep deprivation accelerates aging",
            "water": "3-4 liters daily for maximum skin elasticity",
            "tips": [
                "Start retinol slowly (2x/week) then increase gradually",
                "ALWAYS use sunscreen - sun damage is #1 cause of wrinkles",
                "Sleep on your back to avoid compression wrinkles",
                "Consider professional treatments: Botox, Fillers, Microneedling, RF therapy",
                "Facial exercises can help tone muscles",
                "Quit smoking - it drastically accelerates skin aging",
                "Use humidifier at night for skin hydration",
                "Supplement: Collagen peptides, Vitamin E, Omega-3"
            ]
        }
    }
    
    return recommendations.get(feature, recommendations["clear face"])

# ----------------------------------------------------------
# 📹 WEBCAM VIDEO PROCESSOR CLASS
# ----------------------------------------------------------
class LiveSkinDetector(VideoProcessorBase):
    def __init__(self):
        self.result_text = "Initializing..."
        self.confidence = 0
        self.age = 0
        self.latest_frame = None
        self.annotated_frame = None
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img.copy()
        
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(rgb)
            
            if len(detections) > 0:
                x, y, w, h = detections[0]['box']
                x, y = max(0, x), max(0, y)
                
                cv2.rectangle(img, (x, y), (x+w, y+h), (155, 80, 220), 3)
                
                face = rgb[y:y+h, x:x+w]
                if face.size > 0:
                    face_resized = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
                    face_array = preprocess_input(np.expand_dims(img_to_array(face_resized), axis=0))
                    
                    preds = model.predict(face_array, verbose=0)[0]
                    class_idx = np.argmax(preds)
                    predicted_class = classes[class_idx]
                    confidence = float(preds[class_idx]) * 100
                    
                    # Single stable age estimate
                    age_ranges = {
                        "clear face": (18, 30),
                        "darkspots": (30, 40),
                        "puffy eyes": (40, 55),
                        "wrinkles": (60, 75)
                    }
                    min_age, max_age = age_ranges[predicted_class]
                    est_age = int((min_age + max_age) / 2)
                    
                    self.result_text = predicted_class
                    self.confidence = confidence
                    self.age = est_age
                    
                    label = f"{predicted_class.upper()}"
                    conf_label = f"Confidence: {confidence:.1f}%"
                    age_label = f"Est. Age: {est_age}"
                    
                    # Draw labels with background
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img, (x, y-80), (x+text_w+20, y-10), (75, 0, 100), -1)
                    
                    cv2.putText(img, label, (x+5, y-50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(img, conf_label, (x+5, y-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
                    cv2.putText(img, age_label, (x+5, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)
            else:
                self.result_text = "No face detected"
                cv2.putText(img, "Position face in frame", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            self.annotated_frame = img.copy()
                           
        except Exception as e:
            self.result_text = f"Error: {str(e)}"
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------------------------------------
# 🧠 HEADER
# ----------------------------------------------------------
st.markdown("<h1>🔬 DermalScan AI</h1>", unsafe_allow_html=True)
st.markdown("<h4>Advanced Skin Analysis with Personalized Care Recommendations</h4>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------------------------------------
# 📋 MODE SELECTION
# ----------------------------------------------------------
mode = st.radio(
    "Choose Input Mode:",
    ["📤 Upload Image", "📹 Live Webcam Capture"],
    horizontal=True
)

st.markdown("---")

# ----------------------------------------------------------
# ⚙️ UNIFIED PROCESSING FUNCTION
# ----------------------------------------------------------
def process_and_display_results(image, is_webcam=False):
    """Process image and display comprehensive results with recommendations"""
    
    start_time = time.time()
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)

    if len(detections) == 0:
        h, w, _ = image.shape
        detections = [{'box': (0, 0, w, h)}]

    results, coords = [], []
    used_positions = []

    for det in detections:
        x, y, w, h = det['box']
        x, y = max(0, x), max(0, y)
        face = rgb[y:y + h, x:x + w]
        if face.size == 0:
            continue

        face_resized = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
        face_array = preprocess_input(np.expand_dims(img_to_array(face_resized), axis=0))
        preds = model.predict(face_array, verbose=0)[0]
        class_idx = np.argmax(preds)
        predicted_class = classes[class_idx]
        confidence = float(preds[class_idx]) * 100

        # Stable age estimation
        age_ranges = {
            "clear face": (18, 30),
            "darkspots": (30, 40),
            "puffy eyes": (40, 55),
            "wrinkles": (60, 75)
        }
        min_age, max_age = age_ranges[predicted_class]
        est_age = int((min_age + max_age) / 2)

        results.append({
            "Feature": predicted_class,
            "Confidence (%)": round(confidence, 1),
            "Estimated Age": est_age
        })
        coords.append((x, y, w, h))

        cv2.rectangle(image, (x, y), (x + w, y + h), (155, 80, 220), 2)

        label = f"{predicted_class[:10]}  {confidence:.1f}%  |  Age {est_age}"
        text_x = x
        text_y = y - 8

        for prev_y in used_positions:
            if abs(text_y - prev_y) < 25:
                text_y += 25
        used_positions.append(text_y)

        if text_y < 15:
            text_y = y + h + 18

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        bg_start = (text_x, text_y - text_h - 2)
        bg_end = (text_x + text_w + 4, text_y + 4)
        cv2.rectangle(image, bg_start, bg_end, (75, 0, 100), -1)
        cv2.putText(image, label, (text_x + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 235, 255), 1, cv2.LINE_AA)

    latency = time.time() - start_time
    results_df = pd.DataFrame(results)
    
    return image, results_df, coords, latency

# ----------------------------------------------------------
# 📹 WEBCAM MODE
# ----------------------------------------------------------
if mode == "📹 Live Webcam Capture":
    st.markdown("### 📸 Live Skin Detection & Snapshot")
    st.markdown("<p style='color:#E8D6F0;'>Allow camera access, position your face, then capture when ready</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ctx = webrtc_streamer(
            key="skin-detection-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=LiveSkinDetector,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.markdown("### 📊 Live Preview")
        result_placeholder = st.empty()
        
        if ctx.video_processor:
            result_placeholder.markdown(f"""
            <div class='recommendation-box'>
                <h4 style='color:#9D4EDD;margin-top:0;'>Current Detection:</h4>
                <p style='font-size:1.1em;'><b>Feature:</b> {ctx.video_processor.result_text}</p>
                <p style='font-size:1.1em;'><b>Confidence:</b> {ctx.video_processor.confidence:.1f}%</p>
                <p style='font-size:1.1em;'><b>Estimated Age:</b> {ctx.video_processor.age} years</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Capture button
    if st.button("📸 CAPTURE & ANALYZE", type="primary", use_container_width=True):
        if ctx.video_processor and ctx.video_processor.annotated_frame is not None:
            captured_frame = ctx.video_processor.annotated_frame.copy()
            
            with st.spinner("🔬 Analyzing captured image..."):
                annotated_img, results_df, coords, latency = process_and_display_results(captured_frame, is_webcam=True)
            
            st.success(f"✅ Analysis Completed in {latency:.2f} seconds")
            
            # Store in session state for download
            st.session_state['captured_image'] = annotated_img
            st.session_state['results_df'] = results_df
            st.session_state['coords'] = coords
            st.session_state['latency'] = latency
            st.session_state['capture_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            st.error("❌ No frame captured. Please ensure webcam is active.")
    
    # Display results if capture exists
    if 'captured_image' in st.session_state:
        st.markdown("---")
        st.markdown("## 📋 Detailed Analysis Report")
        
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.image(cv2.cvtColor(st.session_state['captured_image'], cv2.COLOR_BGR2RGB),
                     caption=f"🖼️ Captured at {st.session_state['capture_time']}", 
                     use_container_width=True)
            
            # Download annotated image
            _, buf = cv2.imencode(".jpg", st.session_state['captured_image'])
            st.download_button(
                "📸 Download Annotated Image", 
                data=buf.tobytes(),
                file_name=f"DermalScan_Webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", 
                mime="image/jpeg",
                use_container_width=True
            )
        
        with col2:
            st.markdown("### 📊 Analysis Summary")
            
            if not st.session_state['results_df'].empty:
                detailed_results = []
                for (x, y, w, h), row in zip(st.session_state['coords'], 
                                             st.session_state['results_df'].to_dict(orient="records")):
                    detailed_results.append({
                        "X": x, "Y": y, "Width": w, "Height": h,
                        "Feature": row["Feature"],
                        "Confidence (%)": row["Confidence (%)"],
                        "Estimated Age": row["Estimated Age"]
                    })
                summary_df = pd.DataFrame(detailed_results)

                st.markdown("""
                <style>
                .big-table table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 1.05rem !important;
                    color: #F8F8FF !important;
                }
                .big-table th {
                    background-color: #5B1B3B;
                    padding: 10px;
                    text-align: left;
                    color: #EFBBD2 !important;
                    border-bottom: 2px solid #9A4D7F;
                }
                .big-table td {
                    background-color: rgba(155,55,155,0.15);
                    border-bottom: 1px solid #9A4D7F;
                    padding: 8px 10px;
                }
                </style>
                """, unsafe_allow_html=True)

                st.markdown("<div class='big-table'>", unsafe_allow_html=True)
                st.write(summary_df)
                st.markdown("</div>", unsafe_allow_html=True)

                avg_age = st.session_state['results_df']["Estimated Age"].mean()
                avg_conf = st.session_state['results_df']["Confidence (%)"].mean()
                
                st.markdown(f"<p>🧬 <span style='color:#9D4EDD;font-weight:600;'>Avg Biological Age:</span> {avg_age:.1f} years</p>", unsafe_allow_html=True)
                st.markdown(f"<p>🎯 <span style='color:#9D4EDD;font-weight:600;'>Avg Confidence:</span> {avg_conf:.1f}%</p>", unsafe_allow_html=True)
                st.markdown(f"<p>⏱️ <span style='color:#9D4EDD;font-weight:600;'>Analysis Time:</span> {st.session_state['latency']:.2f}s</p>", unsafe_allow_html=True)

                # Download CSV
                csv = summary_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Report (CSV)", 
                    data=csv,
                    file_name=f"DermalScan_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Personalized Recommendations
        st.markdown("---")
        st.markdown("## 💆‍♀️ Personalized Skincare Recommendations")
        
        primary_feature = st.session_state['results_df'].iloc[0]['Feature']
        primary_conf = st.session_state['results_df'].iloc[0]['Confidence (%)']
        primary_age = st.session_state['results_df'].iloc[0]['Estimated Age']
        
        rec = get_skincare_recommendations(primary_feature, primary_conf, primary_age)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class='recommendation-box'>
                <h3 style='color:#9D4EDD;'>🧴 Daily Routine</h3>
                {''.join([f"<p><b>{i+1}.</b> {routine}</p>" for i, routine in enumerate(rec['routine'])])}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='recommendation-box'>
                <h3 style='color:#9D4EDD;'>🛒 Recommended Products</h3>
                {''.join([f"<p>• {product}</p>" for product in rec['products']])}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='recommendation-box'>
                <h3 style='color:#9D4EDD;'>😴 Sleep & Hydration</h3>
                <p><b>Sleep:</b> {rec['sleep']}</p>
                <p><b>Water Intake:</b> {rec['water']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='recommendation-box'>
                <h3 style='color:#9D4EDD;'>💡 Expert Tips</h3>
                {''.join([f"<p>• {tip}</p>" for tip in rec['tips']])}
            </div>
            """, unsafe_allow_html=True)

# ----------------------------------------------------------
# 📤 UPLOAD IMAGE MODE
# ----------------------------------------------------------
else:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        uploaded_file = st.file_uploader("📤 Upload Your Face Image", type=["jpg", "jpeg", "png"])
        st.markdown("<p style='color:#E8D6F0;'>Ensure good lighting and a front-facing pose.</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='recommendation-box'><b>Supported Detections:</b><br>• Wrinkles<br>• Dark Spots<br>• Puffy Eyes<br>• Clear Skin</div>", unsafe_allow_html=True)

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        with st.spinner("🔬 Running AI Scan..."):
            annotated_img, results_df, coords, latency = process_and_display_results(image)

        st.success(f"✅ Scan Completed in {latency:.2f} seconds")

        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                     caption="🖼️ AI-Annotated Result", use_container_width=True)
            _, buf = cv2.imencode(".jpg", annotated_img)
            st.download_button("📸 Download Annotated Image", data=buf.tobytes(),
                               file_name="DermalScan_Annotated.jpg", mime="image/jpeg",
                               use_container_width=True)

        with col2:
            st.markdown("### 📊 Analysis Summary")
            if not results_df.empty:
                detailed_results = []
                for (x, y, w, h), row in zip(coords, results_df.to_dict(orient="records")):
                    detailed_results.append({
                        "X": x, "Y": y, "Width": w, "Height": h,
                        "Feature": row["Feature"],
                        "Confidence (%)": row["Confidence (%)"],
                        "Estimated Age": row["Estimated Age"]
                    })
                summary_df = pd.DataFrame(detailed_results)

                st.markdown("""
                <style>
                .big-table table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 1.05rem !important;
                    color: #F8F8FF !important;
                }
                .big-table th {
                    background-color: #5B1B3B;
                    padding: 10px;
                    text-align: left;
                    color: #EFBBD2 !important;
                    border-bottom: 2px solid #9A4D7F;
                }
                .big-table td {
                    background-color: rgba(155,55,155,0.15);
                    border-bottom: 1px solid #9A4D7F;
                    padding: 8px 10px;
                }
                </style>
                """, unsafe_allow_html=True)

                st.markdown("<div class='big-table'>", unsafe_allow_html=True)
                st.write(summary_df)
                st.markdown("</div>", unsafe_allow_html=True)

                avg_age = results_df["Estimated Age"].mean()
                avg_conf = results_df["Confidence (%)"].mean()
                
                st.markdown(f"<p>🧬 <span style='color:#9D4EDD;font-weight:600;'>Avg Biological Age:</span> {avg_age:.1f} years</p>", unsafe_allow_html=True)
                st.markdown(f"<p>🎯 <span style='color:#9D4EDD;font-weight:600;'>Avg Confidence:</span> {avg_conf:.1f}%</p>", unsafe_allow_html=True)
                st.markdown(f"<p>⏱️ <span style='color:#9D4EDD;font-weight:600;'>Analysis Time:</span> {latency:.2f}s</p>", unsafe_allow_html=True)

                csv = summary_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Report (CSV)", data=csv,
                                   file_name="DermalScan_Report.csv", mime="text/csv",
                                   use_container_width=True)

                with open("DermalScan_Logs.txt", "a") as log:
                    log.write(f"\n--- Scan @ {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                    log.write(summary_df.to_string(index=False))
                    log.write(f"\nAverage Age: {avg_age:.1f} | Confidence: {avg_conf:.1f}% | Time: {latency:.2f}s\n")

            else:
                st.warning("⚠️ No face detected in the image.")
        
        # Personalized Recommendations for Upload Mode
        if not results_df.empty:
            st.markdown("---")
            st.markdown("## 💆‍♀️ Personalized Skincare Recommendations")
            
            primary_feature = results_df.iloc[0]['Feature']
            primary_conf = results_df.iloc[0]['Confidence (%)']
            primary_age = results_df.iloc[0]['Estimated Age']
            
            rec = get_skincare_recommendations(primary_feature, primary_conf, primary_age)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class='recommendation-box'>
                    <h3 style='color:#9D4EDD;'>🧴 Daily Routine</h3>
                    {''.join([f"<p><b>{i+1}.</b> {routine}</p>" for i, routine in enumerate(rec['routine'])])}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='recommendation-box'>
                    <h3 style='color:#9D4EDD;'>🛒 Recommended Products</h3>
                    {''.join([f"<p>• {product}</p>" for product in rec['products']])}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='recommendation-box'>
                    <h3 style='color:#9D4EDD;'>😴 Sleep & Hydration</h3>
                    <p><b>Sleep:</b> {rec['sleep']}</p>
                    <p><b>Water Intake:</b> {rec['water']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='recommendation-box'>
                    <h3 style='color:#9D4EDD;'>💡 Expert Tips</h3>
                    {''.join([f"<p>• {tip}</p>" for tip in rec['tips']])}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👆 Upload a facial image to begin the AI scan.")

# ----------------------------------------------------------
# 🧩 FOOTER
# ----------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#9D4EDD; font-size:0.9em;'>
⚙️ DermalScan v4.0 | Advanced Skin Analysis & Personalized Care | Developed by Shreya
</div>
""", unsafe_allow_html=True)