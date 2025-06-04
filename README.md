# Real-Time Emotion Detection App

This project is a Streamlit application that performs real-time emotion detection using a PyTorch model based on AlexNet.
The app captures frames from your webcam, runs the model on each frame and updates a bar graph showing the distribution of detected emotions.

The model file (`alexnet_full_model.pth`) is not included in this repository.
After cloning, place your trained model at `models/alexnet_full_model.pth`.

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
