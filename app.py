import streamlit as st
from transformers import pipeline
from pytube import YouTube

# Laden des Whisper-Modells
@st.cache_resource
def load_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-large-v2", device=0)

model = load_model()

# Streamlit-UI
st.title("Arabische Transkriptions-App")
st.write("Nutze OpenAI Whisper Large v2, um arabische Audiodateien oder YouTube-Videos zu transkribieren.")

# Auswahlmöglichkeit: Datei-Upload oder YouTube-URL
option = st.radio("Wähle eine Option:", ["Audio-Datei hochladen", "YouTube-URL eingeben"])

if option == "Audio-Datei hochladen":
    uploaded_file = st.file_uploader("Lade eine Audiodatei hoch (mp3, wav)", type=["mp3", "wav"])
    if uploaded_file is not None:
        st.write("Die Datei wird verarbeitet...")
        transcription = model(uploaded_file)["text"]
        st.write("**Transkription:**")
        st.text(transcription)

elif option == "YouTube-URL eingeben":
    youtube_url = st.text_input("Füge die YouTube-URL ein:")
    if youtube_url:
        try:
            st.write("Das Audio wird vom Video extrahiert...")
            video = YouTube(youtube_url)
            audio_stream = video.streams.filter(only_audio=True).first()
            audio_path = audio_stream.download(filename="temp_audio.mp4")
            transcription = model(audio_path)["text"]
            st.write("**Transkription:**")
            st.text(transcription)
        except Exception as e:
            st.error(f"Fehler: {str(e)}")
