import streamlit as st
# import cv2
import numpy as np
import PIL.Image
from PIL import Image
import io
import time
import os
import math
import requests
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🔍",
    layout="wide"
)

# Configuration flags
USE_DNN_DETECTOR = True       # Toggle between Haar Cascade and DNN detector
ENABLE_EMOTION_DETECTION = True  # Enable emotion detection
ENABLE_FRIEND_INFO = True     # Enable friend information display
ENABLE_ANIMATIONS = True      # Enable animation effects

# Animation settings
PULSE_SPEED = 2.0             # Speed of pulsing animations
SLIDE_SPEED = 15              # Speed of sliding animations (pixels per frame)
FADE_SPEED = 0.1              # Speed of fade-in animations (alpha per frame)

# Load emotion labels
emotion_labels = ['neutral', 'happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'contempt']

# Friend profiles with expanded information for ID card display
friend_profiles = {
    "friend1": {
        "name": "Zohaib Ali Dayo",
        "age": 21,
        "category": "Close Friend",
        "id": "FR-2025-001",
        "since": "2018",
        "interests": "Hacking, Coding",
        "relationship_score": 95,
        "image": "alice_profile.jpg"  # This will be a placeholder
    }
}

# Add more friends for demo purposes
friend_profiles["friend2"] = {
    "name": "John Smith",
    "age": 25,
    "category": "Work Colleague",
    "id": "FR-2025-002",
    "since": "2023",
    "interests": "AI, Data Science",
    "relationship_score": 80,
    "image": "john_profile.jpg"
}

# Add model download URLs
files_to_download = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "emotion-ferplus-8.onnx": "https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
}

# Download function for models
@st.cache_resource
def download_models():
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    for filename, url in files_to_download.items():  # Now works correctly
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath):
            with st.spinner(f"Downloading {filename}..."):
                response = requests.get(url)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                st.success(f"Downloaded {filename}")
    
    return models_dir

# Initialize face detector
@st.cache_resource
def initialize_face_detector(models_dir):
    if USE_DNN_DETECTOR:
        face_detector_net = cv2.dnn.readNetFromCaffe(
            f"{models_dir}/deploy.prototxt",
            f"{models_dir}/res10_300x300_ssd_iter_140000.caffemodel"
        )
    else:
        face_detector_net = None
        
    return face_detector_net

# Initialize emotion detector
@st.cache_resource
def initialize_emotion_detector(models_dir):
    if ENABLE_EMOTION_DETECTION:
        emotion_path = f"{models_dir}/emotion-ferplus-8.onnx"
        if os.path.exists(emotion_path):
            emotion_net = cv2.dnn.readNetFromONNX(emotion_path)
            return emotion_net
    return None

# Convert a NumPy image (RGBA) to a base64 PNG for overlay
def rgba_to_base64(rgba_array):
    img = PIL.Image.fromarray(rgba_array, 'RGBA')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Draw stylized ID card for friend information
def draw_id_card(overlay, x, y, w, h, friend_info, frame_count):
    # Card dimensions and position (below the face)
    card_width = max(300, w * 1.5)
    card_height = 160
    card_x = max(10, min(x - (card_width - w) // 2, overlay.shape[1] - card_width - 10))
    card_y = y + h + 20

    # Animation effects
    if ENABLE_ANIMATIONS:
        # Calculate animation progress (0.0 to 1.0)
        alpha = min(1.0, frame_count * FADE_SPEED)

        # Slide-in animation
        slide_offset = max(0, int((1.0 - alpha) * 50))
        card_y += slide_offset

        # Border pulse effect
        pulse_factor = math.sin(frame_count * 0.1 * PULSE_SPEED) * 0.5 + 0.5
        border_b = int(100 + 155 * pulse_factor)
        border_color = (255, 200, border_b, int(255 * alpha))
    else:
        alpha = 1.0
        border_color = (255, 200, 100, 255)

    # Draw main card background with rounded corners
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_width, card_y + card_height),
                 (50, 50, 70, int(220 * alpha)), -1)

    # Draw stylish border
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_width, card_y + card_height),
                 border_color, 3)

    # Draw header strip
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_width, card_y + 30),
                 (70, 130, 180, int(240 * alpha)), -1)

    # Draw "FRIEND ID" text in header
    cv2.putText(overlay, "FRIEND ID", (card_x + 10, card_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255, int(255 * alpha)), 2)

    # Draw ID number in header
    cv2.putText(overlay, friend_info['id'], (card_x + card_width - 110, card_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, int(255 * alpha)), 1)

    # Draw name with larger font
    cv2.putText(overlay, friend_info['name'], (card_x + 15, card_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, int(255 * alpha)), 2)

    # Draw category with badge-like background
    category_width = 10 + len(friend_info['category']) * 10
    cv2.rectangle(overlay, (card_x + 15, card_y + 70),
                 (card_x + 15 + category_width, card_y + 90),
                 (100, 180, 100, int(200 * alpha)), -1)
    cv2.putText(overlay, friend_info['category'], (card_x + 20, card_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, int(255 * alpha)), 1)

    # Draw additional info
    cv2.putText(overlay, f"Age: {friend_info['age']}", (card_x + 15, card_y + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200, int(255 * alpha)), 1)
    cv2.putText(overlay, f"Since: {friend_info['since']}", (card_x + 15, card_y + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200, int(255 * alpha)), 1)

    # Draw relationship score bar
    score = friend_info['relationship_score']
    bar_width = int((card_width - 160) * (score / 100))
    cv2.putText(overlay, "Relationship:", (card_x + 15, card_y + 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200, int(255 * alpha)), 1)
    cv2.rectangle(overlay, (card_x + 120, card_y + 143),
                 (card_x + card_width - 25, card_y + 153),
                 (80, 80, 80, int(200 * alpha)), -1)

    # Color gradient based on score
    if score >= 90:
        bar_color = (50, 200, 50, int(230 * alpha))
    elif score >= 70:
        bar_color = (50, 180, 200, int(230 * alpha))
    else:
        bar_color = (180, 180, 50, int(230 * alpha))

    cv2.rectangle(overlay, (card_x + 120, card_y + 143),
                 (card_x + 120 + bar_width, card_y + 153),
                 bar_color, -1)

    # Add score text
    cv2.putText(overlay, f"{score}%", (card_x + card_width - 45, card_y + 152),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, int(255 * alpha)), 1)

# Helper function to draw animated face rectangle
def draw_animated_face_box(overlay, x, y, w, h, emotion, frame_count):
    if ENABLE_ANIMATIONS:
        # Pulsing effect for face box based on emotion
        pulse = math.sin(frame_count * 0.1 * PULSE_SPEED) * 0.5 + 0.5

        # Different colors for different emotions
        if emotion in ['happy', 'surprise']:
            color = (0, int(180 + 75 * pulse), int(100 + 155 * pulse), 255)
        elif emotion in ['sad', 'anger', 'disgust', 'fear', 'contempt']:
            color = (int(100 + 155 * pulse), 0, int(180 + 75 * pulse), 255)
        else:  # neutral
            color = (int(100 + 155 * pulse), int(100 + 155 * pulse), 0, 255)

        # Draw the main rectangle
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        # Draw animated corner brackets
        bracket_length = int(min(w, h) * 0.2)
        thickness = 3

        # Top-left corner
        cv2.line(overlay, (x, y), (x + bracket_length, y), color, thickness)
        cv2.line(overlay, (x, y), (x, y + bracket_length), color, thickness)

        # Top-right corner
        cv2.line(overlay, (x + w, y), (x + w - bracket_length, y), color, thickness)
        cv2.line(overlay, (x + w, y), (x + w, y + bracket_length), color, thickness)

        # Bottom-left corner
        cv2.line(overlay, (x, y + h), (x + bracket_length, y + h), color, thickness)
        cv2.line(overlay, (x, y + h), (x, y + h - bracket_length), color, thickness)

        # Bottom-right corner
        cv2.line(overlay, (x + w, y + h), (x + w - bracket_length, y + h), color, thickness)
        cv2.line(overlay, (x + w, y + h), (x + w, y + h - bracket_length), color, thickness)
    else:
        # Simple rectangle if animations disabled
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0, 255), 2)

def process_frame(img, face_detector_net, emotion_net, frame_count, current_friend_key):
    # Create an overlay for face bounding boxes and emotion labels
    bbox_array = np.zeros([img.shape[0], img.shape[1], 4], dtype=np.uint8)
    
    # Face detection using DNN (or Haar Cascade if toggled)
    if USE_DNN_DETECTOR:
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        face_detector_net.setInput(blob)
        detections = face_detector_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                x, y = max(0, x), max(0, y)
                x2, y2 = min(w, x2), min(h, y2)
                faces.append((x, y, x2 - x, y2 - y))
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray)

    # Process each detected face
    face_count = len(faces)
    for i, (x, y, w_face, h_face) in enumerate(faces):
        # Detect emotion
        detected_emotion = "neutral"
        if ENABLE_EMOTION_DETECTION and emotion_net is not None:
            try:
                face_roi = img[y:y+h_face, x:x+w_face]
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
                resized_face = cv2.resize(gray_face, (64, 64))
                normalized_face = resized_face.astype(np.float32) / 255.0
                blob = cv2.dnn.blobFromImage(normalized_face)
                emotion_net.setInput(blob)
                emotions = emotion_net.forward()
                emotion_id = np.argmax(emotions)
                detected_emotion = emotion_labels[emotion_id]

                # Draw emotion with animated background
                if ENABLE_ANIMATIONS:
                    # Draw animated emotion badge
                    text_size = cv2.getTextSize(detected_emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    badge_width = text_size[0] + 20
                    badge_height = text_size[1] + 10
                    badge_x = x + (w_face - badge_width) // 2
                    badge_y = y - badge_height - 5

                    # Ensure badge stays within frame
                    badge_x = max(5, min(badge_x, img.shape[1] - badge_width - 5))
                    badge_y = max(5, badge_y)

                    # Emotion-specific colors
                    if detected_emotion == 'happy':
                        emotion_color = (50, 200, 50, 230)
                    elif detected_emotion == 'sad':
                        emotion_color = (50, 50, 200, 230)
                    elif detected_emotion in ['anger', 'disgust', 'contempt']:
                        emotion_color = (50, 50, 180, 230)
                    elif detected_emotion == 'surprise':
                        emotion_color = (180, 100, 50, 230)
                    else:  # neutral, fear
                        emotion_color = (120, 120, 120, 230)

                    # Pulsing background
                    pulse = math.sin(frame_count * 0.1 * PULSE_SPEED) * 0.3 + 0.7
                    alpha_pulse = int(emotion_color[3] * pulse)

                    # Draw rounded badge background
                    cv2.rectangle(bbox_array,
                               (badge_x, badge_y),
                               (badge_x + badge_width, badge_y + badge_height),
                               (emotion_color[0], emotion_color[1], emotion_color[2], alpha_pulse), -1)

                    # Draw emotion text
                    cv2.putText(bbox_array, detected_emotion,
                              (badge_x + 10, badge_y + badge_height - 7),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                              (255, 255, 255, 255), 2)
                else:
                    # Simple emotion text if animations disabled
                    cv2.putText(bbox_array, detected_emotion,
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                              (0, 255, 0, 255), 2)
            except Exception as e:
                st.error(f"Emotion detection error: {str(e)}")
                pass

        # Draw animated face box
        draw_animated_face_box(bbox_array, x, y, w_face, h_face, detected_emotion, frame_count)

        # Display friend ID card for the first detected face
        if ENABLE_FRIEND_INFO and i == 0:
            friend_info = friend_profiles[current_friend_key]
            draw_id_card(bbox_array, x, y, w_face, h_face, friend_info, frame_count)

    # Display status info with stylish overlay
    if ENABLE_ANIMATIONS:
        # Draw semi-transparent header bar
        cv2.rectangle(bbox_array, (0, 0), (img.shape[1], 40), (0, 0, 0, 180), -1)

        # Add glowing effect based on face detection
        if face_count > 0:
            glow_intensity = math.sin(frame_count * 0.1 * PULSE_SPEED) * 0.5 + 0.5
            glow_color = (0, int(100 * glow_intensity), int(200 * glow_intensity), 100)
            cv2.rectangle(bbox_array, (0, 0), (img.shape[1], 40), glow_color, -1)
            status = "ACTIVE MONITORING"
        else:
            status = "SEARCHING FOR FACES"

        # Draw status text with drop shadow
        shadow_offset = 2
        cv2.putText(bbox_array, status, (12, 30 + shadow_offset),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0, 180), 2)
        cv2.putText(bbox_array, status, (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2)

        # Display face count and FPS in right corner with shadow
        info_text = f"FACES: {face_count}"
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = img.shape[1] - text_size[0] - 10

        cv2.putText(bbox_array, info_text, (text_x + shadow_offset, 30 + shadow_offset),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0, 180), 2)
        cv2.putText(bbox_array, info_text, (text_x, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255, 255), 2)
    else:
        # Simple status display if animations disabled
        cv2.putText(bbox_array, f'Faces: {face_count}', (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0, 255), 2)

    # Update the alpha channel based on drawn content
    bbox_array[:, :, 3] = (bbox_array.max(axis=2) > 0).astype(int) * 255
    
    # Combine original image with overlay
    # Convert original image to RGBA
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # Alpha blending
    alpha_overlay = bbox_array[:, :, 3] / 255.0
    alpha_img = 1.0 - alpha_overlay
    
    for c in range(0, 3):
        img_rgba[:, :, c] = (alpha_overlay * bbox_array[:, :, c] + 
                            alpha_img * img_rgba[:, :, c])
    
    return img_rgba, face_count

def main():
    # Add a custom title with styling
    st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #4CAF50;
        background-color: #333;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .subtitle {
        text-align: center;
        font-style: italic;
        margin-bottom: 30px;
    }
    .stApp {
        background-color: #222;
        color: #ddd;
    }
    </style>
    <div class="title">
        <h1>Advanced Face Recognition System</h1>
    </div>
    <div class="subtitle">
        <h4>Featuring emotion detection and friend identification</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.title("Controls")
    
    # Display options
    st.sidebar.header("Display Options")
    enable_emotion = st.sidebar.checkbox("Enable Emotion Detection", value=True)
    enable_friend_info = st.sidebar.checkbox("Show Friend ID Card", value=True)
    enable_animations = st.sidebar.checkbox("Enable Animations", value=True)
    
    # Friend selection
    st.sidebar.header("Friend Selection")
    friend_selection = st.sidebar.radio(
        "Select Friend to Display",
        list(friend_profiles.keys()),
        format_func=lambda x: friend_profiles[x]["name"]
    )
    
    # Advanced settings
    st.sidebar.header("Advanced Settings")
    detection_confidence = st.sidebar.slider("Detection Confidence", 0.3, 0.9, 0.5, 0.1)
    
    # Update global flags based on user selections
    global ENABLE_EMOTION_DETECTION, ENABLE_FRIEND_INFO, ENABLE_ANIMATIONS
    ENABLE_EMOTION_DETECTION = enable_emotion
    ENABLE_FRIEND_INFO = enable_friend_info
    ENABLE_ANIMATIONS = enable_animations
    
    # Download and initialize models
    models_dir = download_models()
    face_detector_net = initialize_face_detector(models_dir)
    emotion_net = initialize_emotion_detector(models_dir)
    
    # Create two columns for the main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Add a camera feed
        st.header("Camera Feed")
        camera_placeholder = st.empty()
        
        # Add status indicators
        status_placeholder = st.empty()
        
        # Add a start button
        start_button = st.button("Start Face Recognition")
        stop_button = st.button("Stop")
        
    with col2:
        # Display friend information
        st.header("Selected Friend Profile")
        friend_info = friend_profiles[friend_selection]
        
        # Create a styled card for friend info
        st.markdown(f"""
        <div style="background-color:#444; padding:20px; border-radius:10px; border:2px solid #70A9A1;">
            <h3 style="color:#4CAF50; margin-top:0;">{friend_info['name']}</h3>
            <div style="background-color:#2E6171; display:inline-block; padding:5px 10px; border-radius:5px; margin-bottom:10px;">
                {friend_info['category']}
            </div>
            <p><strong>ID:</strong> {friend_info['id']}</p>
            <p><strong>Age:</strong> {friend_info['age']}</p>
            <p><strong>Since:</strong> {friend_info['since']}</p>
            <p><strong>Interests:</strong> {friend_info['interests']}</p>
            <div style="margin-top:10px;">
                <p><strong>Relationship Score:</strong></p>
                <div style="background-color:#333; height:20px; border-radius:5px; width:100%;">
                    <div style="background-color:#4CAF50; height:20px; border-radius:5px; width:{friend_info['relationship_score']}%;">
                    </div>
                </div>
                <p style="text-align:right;">{friend_info['relationship_score']}%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Stats section
        st.header("Recognition Stats")
        stats_placeholder = st.empty()
    
    if start_button:
        if 'frame_count' not in st.session_state:
            st.session_state.frame_count = 0
            st.session_state.face_count_history = []
        
        # Set up the camera
        camera = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not camera.isOpened():
            st.error("Error: Could not open camera. Please check your webcam connection.")
            return
        
        while not stop_button:
            # Read a frame
            ret, frame = camera.read()
            
            if not ret:
                st.error("Failed to capture image from camera")
                break
            
            # Flip the frame horizontally for a more natural view
            frame = cv2.flip(frame, 1)
            
            # Increment frame counter for animations
            st.session_state.frame_count += 1
            
            # Process the frame
            result_frame, face_count = process_frame(
                frame, 
                face_detector_net,
                emotion_net,
                st.session_state.frame_count,
                friend_selection
            )
            
            # Keep track of face counts for stats
            st.session_state.face_count_history.append(face_count)
            if len(st.session_state.face_count_history) > 30:  # Keep last 30 frames
                st.session_state.face_count_history.pop(0)
            
            # Display the processed frame
            camera_placeholder.image(
                result_frame, 
                channels="RGBA",
                use_column_width=True
            )
            
            # Update status
            status_text = "🟢 ACTIVE: Face detection running" if face_count > 0 else "🔍 SEARCHING: No faces detected"
            status_placeholder.markdown(f"<h3 style='color:{'#4CAF50' if face_count > 0 else '#FFA000'}'>{status_text}</h3>", unsafe_allow_html=True)
            
            # Update stats
            avg_faces = sum(st.session_state.face_count_history) / len(st.session_state.face_count_history) if st.session_state.face_count_history else 0
            
            stats_placeholder.markdown(f"""
            <div style="background-color:#333; padding:15px; border-radius:10px;">
                <p><strong>Current faces:</strong> {face_count}</p>
                <p><strong>Average faces:</strong> {avg_faces:.1f}</p>
                <p><strong>Frame count:</strong> {st.session_state.frame_count}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Use a small sleep to reduce CPU usage
            time.sleep(0.05)
        
        # Release the camera when done
        camera.release()
    
    # Add information section at the bottom
    st.markdown("---")
    st.markdown("""
    ### About This Application
    This advanced face recognition system features:
    - Real-time face detection using OpenCV DNN models
    - Emotion recognition to detect user mood
    - Friend identification with animated ID cards
    - Interactive controls and statistics
    
    Built with Streamlit and OpenCV for easy deployment.
    """)
    
if __name__ == "__main__":
    main()
