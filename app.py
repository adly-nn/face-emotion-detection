import streamlit as st
import numpy as np
import sqlite3
from datetime import datetime
from tensorflow import keras
from PIL import Image
import cv2
import os

# ==============================================================
# CONFIGURATION
# ==============================================================

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="😃")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

EMOTION_RESPONSES = {
    'angry': "You look angry. What's bothering you? Take a deep breath! 😤",
    'disgust': "You seem disgusted. Did something unpleasant happen? 🤢",
    'fear': "You look fearful. Everything will be okay! Stay strong! 😨",
    'happy': "You're happy! Keep smiling, it looks great on you! 😊",
    'sad': "You seem sad. Remember, tough times don't last but tough people do! 😢",
    'surprise': "You look surprised! What caught you off guard? 😲",
    'neutral': "You have a neutral expression. Feeling calm today? 😐"
}

# ==============================================================
# DATABASE FUNCTIONS
# ==============================================================

def init_db():
    """Initialize the database and create tables if they don't exist"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            age INTEGER NOT NULL,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_path TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_to_database(name, email, age, emotion, confidence, image_path):
    """Save user data to the database"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO users (name, email, age, emotion, confidence, image_path, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (name, email, age, emotion, confidence, image_path, datetime.now()))
    conn.commit()
    conn.close()

# ==============================================================
# MODEL LOADING
# ==============================================================

@st.cache_resource
def load_emotion_model():
    model = keras.models.load_model('face_emotionModel.h5')
    return model

emotion_model = load_emotion_model()

# ==============================================================
# IMAGE PROCESSING
# ==============================================================

def preprocess_image(image_path):
    """Prepare image for model prediction"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    return normalized.reshape(1, 48, 48, 1)

def predict_emotion(image_path):
    processed = preprocess_image(image_path)
    preds = emotion_model.predict(processed, verbose=0)
    index = np.argmax(preds[0])
    return EMOTIONS[index], float(preds[0][index])

# ==============================================================
# STREAMLIT APP UI
# ==============================================================

st.title("😃 Facial Emotion Recognition Web App")
st.write("Upload your picture to detect your emotion!")

# Initialize database
init_db()

# Input form
with st.form("user_form"):
    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Analyze Emotion")

if submitted:
    if not name or not email or not uploaded_file:
        st.error("⚠️ Please fill in all fields and upload an image.")
    else:
        # Save uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.read())

        # Show uploaded image
        image = Image.open(filepath)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict emotion
        emotion, confidence = predict_emotion(filepath)
        st.subheader(f"Detected Emotion: {emotion.upper()}")
        st.write(f"Confidence: {confidence*100:.2f}%")
        st.info(EMOTION_RESPONSES[emotion])

        # Save to database
        save_to_database(name, email, age, emotion, confidence, filepath)
        st.success("✅ Your result has been saved to the database!")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & TensorFlow")
