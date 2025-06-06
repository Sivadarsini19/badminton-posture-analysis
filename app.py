import cv2
import numpy as np
import streamlit as st
import os
import time
from collections import Counter

def get_openpose_keypoints(output, frame_width, frame_height, threshold=0.1):
    H, W = output.shape[2], output.shape[3]
    points = []
    for i in range(18):
        probMap = output[0, i, :, :]
        _, prob, _, point = cv2.minMaxLoc(probMap)
        x = (frame_width * point[0]) / W
        y = (frame_height * point[1]) / H
        if prob > threshold:
            points.append((int(x), int(y)))
        else:
            points.append(None)
    return points

def classify_stance(points):
    def angle(p1, p2, p3):
        if not all([p1, p2, p3]):
            return None
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-5)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    # Keypoints
    shoulder = points[5]
    hip = points[11]
    knee = points[13]
    ankle = points[15]
    elbow = points[7]
    wrist = points[9]

    spine_angle = angle(shoulder, hip, knee)
    arm_angle = angle(shoulder, elbow, wrist)
    knee_angle = angle(hip, knee, ankle)

    # Classify
    if knee_angle and knee_angle < 120 and spine_angle and spine_angle > 150:
        return "Defensive Stance"
    elif spine_angle and spine_angle < 130 and arm_angle and arm_angle < 150:
        return "Offensive Stance"
    elif spine_angle and 130 <= spine_angle <= 160 and arm_angle and arm_angle > 160:
        return "Ready Position"
    elif spine_angle and spine_angle < 120 and knee_angle and knee_angle > 160:
        return "Backward Attacking Stance"
    else:
        return "Unknown"

# Load OpenPose Model
proto = r"C:\\Users\\sivad\\OneDrive\\Desktop\\review 1 demo\\pose_deploy_linevec.prototxt"
weights = r"C:\\Users\\sivad\\OneDrive\\Desktop\\review 1 demo\\pose_iter_440000.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto, weights)

# Video Path
video_path = r"C:/Users/sivad/OneDrive/Desktop/review 1 demo/badminton1.mp4"
if not os.path.exists(video_path):
    st.error("Video file not found.")
    st.stop()

cap = cv2.VideoCapture(video_path)
st.title("\U0001F3BD Badminton Posture Analysis with OpenPose")
st.text("Real-time frame-by-frame posture detection and final stance summary")

frame_window = st.image([])
st_frame_info = st.empty()
frame_count = 0
total_inference_time = 0

stance_counter = Counter()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w, _ = frame.shape
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)

    start = time.time()
    output = net.forward()
    inference_time = time.time() - start
    total_inference_time += inference_time

    points = get_openpose_keypoints(output, w, h)
    stance = classify_stance(points)
    stance_counter[stance] += 1

    # Draw keypoints
    for p in points:
        if p:
            cv2.circle(frame, p, 5, (0, 255, 0), thickness=-1)

    cv2.putText(frame, f"Inference Time: {inference_time:.2f}s", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    st_frame_info.info(f"Processed Frame: {frame_count} | Inference Time: {inference_time:.2f}s")

cap.release()
st.success("✅ Posture Analysis Completed for All Frames")

# Summary
st.subheader("🏸 Posture Summary for Entire Video")
total_stance_frames = sum(stance_counter.values())
if total_stance_frames > 0:
    for stance, count in stance_counter.items():
        percentage = (count / total_stance_frames) * 100
        st.write(f"- **{stance}**: {count} frames ({percentage:.2f}%)")
    st.bar_chart(stance_counter)
else:
    st.write("No valid postures detected.")

# Overall Analysis
st.subheader("📈 Overall Analysis")
if frame_count > 0:
    average_inference_time = total_inference_time / frame_count
    st.write(f"✅ Total Frames Processed: {frame_count}")
    st.write(f"⏱️ Average Inference Time per Frame: {average_inference_time:.2f} seconds")
else:
    st.warning("No frames were processed.")

# Motivational Feedback
st.subheader("💡 Motivational Feedback and Suggestions")
dominant_stance = stance_counter.most_common(1)[0][0] if stance_counter else "Unknown"

if dominant_stance == "Ready Position":
    st.success("💪 Great job staying alert and balanced in the Ready Position! You're maintaining a solid base for quick reactions.")
    st.info("📌 Tip: Practice transitioning smoothly between Ready and Offensive stance for better court coverage.")
elif dominant_stance == "Defensive Stance":
    st.success("🛡️ Strong defensive form! You’re doing well holding your ground during rallies.")
    st.info("📌 Tip: Work on pushing off your back foot to quickly transition into attack.")
elif dominant_stance == "Offensive Stance":
    st.success("🔥 Impressive aggression in your play! You’re maintaining great momentum during offensive moments.")
    st.info("📌 Tip: Focus on maintaining balance after powerful smashes to recover quickly.")
elif dominant_stance == "Backward Attacking Stance":
    st.success("🔁 Good use of space with Backward Attacking movements!")
    st.info("📌 Tip: Try to reduce back-leaning and keep knees slightly bent for smoother forward transitions.")
else:
    st.warning("🤔 Posture not clearly recognized. Make sure your movements are fully visible to the camera.")
    st.info("📌 Tip: Ensure full-body visibility and good lighting to improve pose detection accuracy.")
