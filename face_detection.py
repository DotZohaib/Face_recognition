# this is real code on introduction
# Import dependencies
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import time
import os
import json
import math  # For animation calculations

# Configuration flags
USE_DNN_DETECTOR = True      # Toggle between Haar Cascade and DNN detector
ENABLE_EMOTION_DETECTION = True # Enable emotion detection
ENABLE_FRIEND_INFO = True    # New flag to enable friend information display
ENABLE_ANIMATIONS = True     # New flag to enable animation effects

# Animation settings
PULSE_SPEED = 2.0            # Speed of pulsing animations
SLIDE_SPEED = 15             # Speed of sliding animations (pixels per frame)
FADE_SPEED = 0.1             # Speed of fade-in animations (alpha per frame)

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
        "image": "/content/alice_profile.jpg"
    }
}

# Select a default friend to display info for
default_friend_name = "friend1"

# Download and load DNN models if needed
if USE_DNN_DETECTOR or ENABLE_EMOTION_DETECTION:
    # Download necessary model files
    !wget -q https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
    !wget -q https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel -O res10_300x300_ssd_iter_140000.caffemodel
    !wget -q https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx

# Initialize face detector
if USE_DNN_DETECTOR:
    face_detector_net = cv2.dnn.readNetFromCaffe('deploy.prototxt',
                                            'res10_300x300_ssd_iter_140000.caffemodel')
else:
    face_cascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades +
                                                                 'haarcascade_frontalface_default.xml'))

# Initialize emotion detector
if ENABLE_EMOTION_DETECTION:
    if os.path.exists('emotion-ferplus-8.onnx'):
        emotion_net = cv2.dnn.readNetFromONNX('emotion-ferplus-8.onnx')
    else:
        print("Error: emotion-ferplus-8.onnx file not found. Please download it.")

# Helper function: Convert JS data URL to an OpenCV image
def js_to_image(js_reply):
    image_bytes = b64decode(js_reply.split(',')[1])
    jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(jpg_as_np, flags=1)

# Helper function: Convert a NumPy image (RGBA) to a base64 PNG for overlay
def bbox_to_bytes(bbox_array):
    bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
    iobuf = io.BytesIO()
    bbox_PIL.save(iobuf, format='png')
    return 'data:image/png;base64,{}'.format(str(b64encode(iobuf.getvalue()), 'utf-8'))

# Helper function: Draw stylized ID card for friend information
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

# JavaScript video stream setup
def video_stream():
    js = Javascript('''
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    var imgElement;
    var labelElement;

    var pendingResolve = null;
    var shutdown = false;

    function removeDom() {
        stream.getVideoTracks()[0].stop();
        video.remove();
        div.remove();
        video = null;
        div = null;
        stream = null;
        imgElement = null;
        captureCanvas = null;
        labelElement = null;
    }

    function onAnimationFrame() {
      if (!shutdown) {
        window.requestAnimationFrame(onAnimationFrame);
      }
      if (pendingResolve) {
        var result = "";
        if (!shutdown) {
          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
          result = captureCanvas.toDataURL('image/jpeg', 0.8)
        }
        var lp = pendingResolve;
        pendingResolve = null;
        lp(result);
      }
    }

   async function createDom() {
      if (div !== null) {
        return stream;
      }

      div = document.createElement('div');
      div.style.border = '2px solid black';
      div.style.padding = '3px';
      div.style.width = '100%';
      div.style.maxWidth = '600px';
      div.style.background = '#333';
      div.style.borderRadius = '8px';
      div.style.boxShadow = '0 4px 8px rgba(0,0,0,0.5)';
      document.body.appendChild(div);

      const modelOut = document.createElement('div');
      modelOut.style.background = '#222';
      modelOut.style.color = '#fff';
      modelOut.style.padding = '8px';
      modelOut.style.borderRadius = '4px 4px 0 0';
      modelOut.style.display = 'flex';
      modelOut.style.justifyContent = 'space-between';
      modelOut.innerHTML = "<span style='font-weight:300'>AI Detection Status:</span>";
      labelElement = document.createElement('span');
      labelElement.innerText = 'Initializing...';
      labelElement.style.fontWeight = 'bold';
      labelElement.style.color = '#4CAF50';
      modelOut.appendChild(labelElement);
      div.appendChild(modelOut);

      video = document.createElement('video');
      video.style.display = 'block';
      video.style.width = '100%';
      video.style.borderRadius = '0';
      video.style.transform = 'scaleX(-1)'; // Keep mirror effect for video
      video.setAttribute('playsinline', '');
      video.onclick = () => { shutdown = true; };
      stream = await navigator.mediaDevices.getUserMedia(
          {video: { facingMode: "user"}});
      div.appendChild(video);

      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute';
      imgElement.style.zIndex = 1;
      // Remove this line or set to 'none' to prevent mirroring the overlay
      imgElement.style.transform = 'none';
      imgElement.onclick = () => { shutdown = true; };
      div.appendChild(imgElement);

      const instruction = document.createElement('div');
      instruction.style.background = '#222';
      instruction.style.color = '#fff';
      instruction.style.padding = '8px';
      instruction.style.borderRadius = '0 0 4px 4px';
      instruction.style.fontSize = '14px';
      instruction.style.textAlign = 'center';
      instruction.innerHTML =
          '<span style="color: #FF5722; font-weight: bold;">' +
          'Click anywhere on the video to stop</span>';
      div.appendChild(instruction);
      instruction.onclick = () => { shutdown = true; };

      video.srcObject = stream;
      await video.play();

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 640;
      captureCanvas.height = 480;
      window.requestAnimationFrame(onAnimationFrame);

      return stream;
    }
    async function stream_frame(label, imgData) {
      if (shutdown) {
        removeDom();
        shutdown = false;
        return '';
      }

      var preCreate = Date.now();
      stream = await createDom();

      var preShow = Date.now();
      if (label != "") {
        labelElement.innerHTML = label;
      }

      if (imgData != "") {
        var videoRect = video.getClientRects()[0];
        imgElement.style.top = videoRect.top + "px";
        imgElement.style.left = videoRect.left + "px";
        imgElement.style.width = videoRect.width + "px";
        imgElement.style.height = videoRect.height + "px";
        imgElement.src = imgData;
      }

      var preCapture = Date.now();
      var result = await new Promise(function(resolve, reject) {
        pendingResolve = resolve;
      });
      shutdown = false;

      return {'create': preShow - preCreate,
              'show': preCapture - preShow,
              'capture': Date.now() - preCapture,
              'img': result};
    }
    ''')
    display(js)

# Helper function to call the JavaScript stream_frame function using JSON encoding
def video_frame(label, bbox):
    # Using json.dumps to safely pass the strings to JavaScript
    return eval_js("stream_frame(%s, %s)" % (json.dumps(label), json.dumps(bbox)))

# Start the video stream
video_stream()

# Give the JavaScript a moment to load properly
time.sleep(2)

label_html = 'Initializing facial recognition system...'
bbox = ''
prev_time = time.time()
face_count = 0
frame_count = 0  # For animation timing

current_friend_index = 0
friend_keys = list(friend_profiles.keys())
friend_switch_interval = 100  # Switch displayed friend every 100 frames

while True:
    # Get video frame from JavaScript
    js_reply = video_frame(label_html, bbox)
    if not js_reply:
        break

    # Increment frame counter
    frame_count += 1

    # Auto-switch friend profile periodically
    if frame_count % friend_switch_interval == 0 and len(friend_keys) > 1:
        current_friend_index = (current_friend_index + 1) % len(friend_keys)

    current_friend_key = friend_keys[current_friend_index]

    # Convert the JS response to an OpenCV image
    img = js_to_image(js_reply["img"])

    # Flip horizontally to match mirrored display in browser
    img = cv2.flip(img, 1)

    # Create an overlay for face bounding boxes and emotion labels
    bbox_array = np.zeros([480, 640, 4], dtype=np.uint8)

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
        faces = face_cascade.detectMultiScale(gray)

    # Process each detected face
    face_count = len(faces)
    for i, (x, y, w_face, h_face) in enumerate(faces):
        # Detect emotion
        detected_emotion = "neutral"
        if ENABLE_EMOTION_DETECTION:
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
                pass

        # Draw animated face box
        draw_animated_face_box(bbox_array, x, y, w_face, h_face, detected_emotion, frame_count)

        # Display friend ID card for the first detected face
        if ENABLE_FRIEND_INFO and i == 0:
            friend_info = friend_profiles[current_friend_key]
            draw_id_card(bbox_array, x, y, w_face, h_face, friend_info, frame_count)

    # Calculate frames per second (FPS)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display status info with stylish overlay
    if ENABLE_ANIMATIONS:
        # Draw semi-transparent header bar
        cv2.rectangle(bbox_array, (0, 0), (640, 40), (0, 0, 0, 180), -1)

        # Add glowing effect based on face detection
        if face_count > 0:
            glow_intensity = math.sin(frame_count * 0.1 * PULSE_SPEED) * 0.5 + 0.5
            glow_color = (0, int(100 * glow_intensity), int(200 * glow_intensity), 100)
            cv2.rectangle(bbox_array, (0, 0), (640, 40), glow_color, -1)
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
        info_text = f"FACES: {face_count} | FPS: {fps:.1f}"
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = 640 - text_size[0] - 10

        cv2.putText(bbox_array, info_text, (text_x + shadow_offset, 30 + shadow_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0, 180), 2)
        cv2.putText(bbox_array, info_text, (text_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255, 255), 2)
    else:
        # Simple status display if animations disabled
        cv2.putText(bbox_array, f'Faces: {face_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0, 255), 2)
        cv2.putText(bbox_array, f'FPS: {fps:.1f}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0, 255), 2)

    # Update label with status
    if face_count > 0:
        label_html = f'<span style="color:#4CAF50">Detected {face_count} face(s)</span>'
    else:
        label_html = '<span style="color:#FFA000">Searching for faces...</span>'

    # Update the alpha channel based on drawn content and convert overlay to data URL
    bbox_array[:, :, 3] = (bbox_array.max(axis=2) > 0).astype(int) * 255
    bbox = bbox_to_bytes(bbox_array)
