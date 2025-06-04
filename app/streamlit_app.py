import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from deepface import DeepFace
import numpy as np
import pandas as pd
import altair as alt

# Emotion labels from DeepFace
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Store emotion counts
if 'emotion_counts' not in st.session_state:
    st.session_state.emotion_counts = {emotion: 0 for emotion in EMOTIONS}

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.analyzer = DeepFace.build_model('Emotion')

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        try:
            result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result['dominant_emotion']
            if dominant_emotion in st.session_state.emotion_counts:
                st.session_state.emotion_counts[dominant_emotion] += 1
        except Exception as e:
            # Ignore detection errors
            pass
        return frame

st.title("Real-time Emotion Detection")

webrtc_streamer(key="emotion", mode=WebRtcMode.SENDRECV,
               video_processor_factory=EmotionProcessor,
               media_stream_constraints={"video": True, "audio": False})

st.header("Emotion Distribution")
counts = st.session_state.emotion_counts
source = pd.DataFrame({"emotion": list(counts.keys()), "count": list(counts.values())})
chart = alt.Chart(source).mark_bar().encode(x="emotion", y="count", color="emotion")
st.altair_chart(chart, use_container_width=True)
