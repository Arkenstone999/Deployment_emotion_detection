import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import pandas as pd
import altair as alt
import torch
from torchvision import transforms
import cv2
from pathlib import Path

# Emotion labels (update to match your model)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Store emotion counts
if 'emotion_counts' not in st.session_state:
    st.session_state.emotion_counts = {emotion: 0 for emotion in EMOTIONS}

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
            emotion = EMOTIONS[pred.item()]
            if emotion in st.session_state.emotion_counts:
                st.session_state.emotion_counts[emotion] += 1
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
