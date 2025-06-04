# Real-Time Emotion Detection App

This project is a simple Streamlit application that uses a pre-trained emotion recognition model from [DeepFace](https://github.com/serengil/deepface) to detect human emotions from a webcam feed.

The UI displays the live video feed along with a bar graph showing the distribution of detected emotions.

## Requirements

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Running the App

To start the Streamlit server and run the application:

```bash
streamlit run app/streamlit_app.py
```

Allow the browser to access your webcam when prompted. The application will analyze each frame and update the emotion distribution graph in real time.
