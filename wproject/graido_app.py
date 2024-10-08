# gradio_app.py
import gradio as gr
import whisper
import requests

def transcribe(audio_file):
    # Send audio data to Django backend
    try:
        with open(audio_file, 'rb') as f:
            files = {'audio': ('audio.wav', f, 'audio/wav')}
            response = requests.post(
                'http://localhost:8000/transcribe/',
                files=files
            )
        response.raise_for_status()  # Raise an exception for bad status codes
        transcription = response.json().get('transcription', '')
        return transcription
    except requests.RequestException as e:
        return f"Error communicating with the server: {str(e)}"

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Audio Transcription with Whisper"
)
 

if __name__ == "__main__":
    iface.launch()