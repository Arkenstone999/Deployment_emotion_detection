import streamlit as st
import torch
from torchvision import transforms
import torchvision
import cv2
from pathlib import Path
import pandas as pd
import altair as alt
import tempfile

import pickle



ENGAGEMENT_TYPES = ["not engaged", "engaged-positive", "engaged-negative"]


@st.cache_resource
def load_model() -> torch.nn.Module:
    model_path = Path("models/alexnet_full_model.pth")
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}. Please add it and restart.")
        st.stop()

    try:
        model = torch.load(model_path, map_location=torch.device("cpu"))
    except pickle.UnpicklingError:
        torch.serialization.add_safe_globals([torchvision.models.alexnet.AlexNet])
        model = torch.load(
            model_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )

    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    return model


transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def process_video(path: str) -> dict:
    model = load_model()
    counts = {e: 0 for e in ENGAGEMENT_TYPES}

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
            _, pred = torch.max(outputs, 1)
            engagement = ENGAGEMENT_TYPES[pred.item()]
            counts[engagement] += 1

        idx += 1
        if total > 0:
            progress.progress(min(idx / total, 1.0))

    cap.release()
    progress.empty()
    return counts


st.title("Video Engagement Detection")

uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.write("Processing video...")
    counts = process_video(tmp_path)

    st.header("Engagement Distribution")
    df = pd.DataFrame({"engagement": list(counts.keys()), "count": list(counts.values())})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(x="engagement", y="count", color="engagement")
    )
    st.altair_chart(chart, use_container_width=True)

