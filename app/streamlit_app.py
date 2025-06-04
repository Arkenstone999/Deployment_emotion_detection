import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import pandas as pd
import altair as alt
import torch
from torchvision import transforms
import cv2
from pathlib import Path

# Engagement labels produced by the model
ENGAGEMENT_TYPES = [
    'not engaged',
    'engaged-positive',
    'engaged-negative',
]

# Store emotion counts
if 'emotion_counts' not in st.session_state:
    st.session_state.emotion_counts = {e: 0 for e in ENGAGEMENT_TYPES}

@st.cache_resource
def load_model() -> torch.nn.Module:
    model_path = Path('models/alexnet_full_model.pth')
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}. Please add it and restart.")
        st.stop()
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model


class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(tensor)
            _, pred = torch.max(outputs, 1)
            engagement = ENGAGEMENT_TYPES[pred.item()]
            if engagement in st.session_state.emotion_counts:
                st.session_state.emotion_counts[engagement] += 1
        cv2.putText(
            img,
            engagement,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")

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
