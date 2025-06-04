

## Requirements

The application is tested with **Python 3.8** or **Python 3.9**.

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

The app relies on **PyTorch** to run the included model. You do **not** need
TensorFlow or DeepFace unless you modify the code.

To experiment with a DeepFace-based variant, install TensorFlow first:

```bash
pip install tensorflow
```

## Running the App

To start the Streamlit server and run the application:

```bash
streamlit run app/streamlit_app.py
```
