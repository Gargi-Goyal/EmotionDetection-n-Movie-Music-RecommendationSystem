import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser

# Load model and labels
model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Set custom page style
st.set_page_config(page_title="Emotionify 🎵🎬", page_icon="🎭", layout="centered")
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #6c63ff;
        text-align: center;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .stButton>button {
        color: white;
        background-color: #6c63ff;
        border-radius: 10px;
        height: 50px;
        font-size: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.header("🎭 Emotionify - Music and Movie Recommender 🎵🎬")

# Session setup
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Emotion processor class
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)
            pred = label[np.argmax(model.predict(lst))]
            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            np.save("emotion.npy", np.array([pred]))  # Save the detected emotion

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                                landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                                connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# UI inputs
with st.expander("🛠 Customize Your Preferences", expanded=True):
    lang = st.text_input("🌐 Enter Language", placeholder="e.g., English, Hindi, Spanish...")
    singer = st.text_input("🎤 Favorite Singer (Optional)", placeholder="e.g., Arijit Singh, Taylor Swift...")
    choice = st.selectbox("🎯 What would you like to get recommended?", ["Songs", "Movies"])

# Live camera capture
if lang and st.session_state["run"] != "false":
    st.markdown("### 📸 Capturing your emotions...")
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

# Recommend button
btn = st.button("🚀 Recommend Me!")

# Button logic with overrides for sad, fear, and angry
if btn:
    # Check if the emotion is already detected, else show warning to detect emotion first
    try:
        emotion = np.load("emotion.npy")[0]
    except:
        emotion = ""

    if not emotion:
        st.warning("⚡ Please capture your emotion first!")
        st.session_state["run"] = "true"
    else:
        emotion_lower = emotion.lower()

        # Based on emotion, set search query for recommendation
        if choice == "Songs":
            if emotion_lower == "sad":
                search_query = f"{lang} happy uplifting songs {singer}"
            elif emotion_lower == "fear":
                search_query = f"{lang} funny cheerful songs {singer}"
            elif emotion_lower == "angry":
                search_query = f"{lang} calm relaxing peaceful songs {singer}"
            else:
                search_query = f"{lang} {emotion} song {singer}"
        else:  # Movies
            if emotion_lower == "sad":
                search_query = f"{lang} happy feel-good comedy movie"
            elif emotion_lower == "fear":
                search_query = f"{lang} funny light-hearted comedy movie"
            elif emotion_lower == "angry":
                search_query = f"{lang} calm relaxing peaceful movie"
            else:
                search_query = f"{lang} {emotion} mood movie"

        # Open YouTube with the search query
        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")

        # Reset emotion to prevent multiple triggers
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"
