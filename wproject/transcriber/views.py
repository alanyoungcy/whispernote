# Create you
import os
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import gradio as gr
import openai
from openai import OpenAI
import threading
import sounddevice as sd
import numpy as np
import queue
import matplotlib.pyplot as plt
import base64
from io import BytesIO


client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', ), base_url=os.getenv('OPENAI_API_BASE', 'https://app.nextchat.dev/'))
 

# Set OpenAI API Key
# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=os.getenv('OPENAI_API_BASE', 'https://app.nextchat.dev/'))'
# openai.api_base = os.getenv('OPENAI_API_BASE', 'https://app.nextchat.dev/') 
# Global variables
is_recording = False
audio_queue = queue.Queue()
transcription_result = ''

def home(request):
    return render(request, 'transcriber/home.html')

def start_stop_recording():
    global is_recording
    if not is_recording:
        is_recording = True
        threading.Thread(target=record_audio).start()
        return "Recording... Press button again to stop"
    else:
        is_recording = False
        return "Recording stopped"

def record_audio():
    global is_recording
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
        while is_recording:
            sd.sleep(100)

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def transcribe_live():
    global transcription_result
    audio_data = []
    while not audio_queue.empty():
        audio_data.append(audio_queue.get())

    if len(audio_data) > 0:
        audio_np = np.concatenate(audio_data, axis=0).flatten()

        # Convert numpy array to bytes
        audio_bytes = audio_np.tobytes()

        # Use OpenAI's Whisper API
        response = client.audio.transcribe_raw("whisper-1", audio_bytes)
        transcription_result += response.text + ' '

    return transcription_result

def update_waveform():
    audio_data = []
    while not audio_queue.empty():
        audio_data.append(audio_queue.get())
    if len(audio_data) > 0:
        audio_np = np.concatenate(audio_data, axis=0).flatten()

        plt.figure(figsize=(10, 4))
        plt.plot(audio_np)
        plt.title('Live Audio Waveform')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return f"data:image/png;base64,{image_base64}"
    else:
        return None

def gradio_interface(request):
    global transcription_result
    transcription_result = ''

    with gr.Blocks() as demo:
        gr.Markdown("# Real-Time Transcription with Whisper and Django")
        record_button = gr.Button("Start/Stop Recording")
        live_transcription = gr.Textbox(label="Live Transcription")
        waveform = gr.Image(label="Live Audio Waveform")

        record_button.click(start_stop_recording, outputs=record_button)
        demo.load(transcribe_live, inputs=None, outputs=live_transcription, every=2)
        demo.load(update_waveform, inputs=None, outputs=waveform, every=1)

    # Enable queuing
    demo.queue()

    return HttpResponse(demo.launch(prevent_thread_lock=True, share=True))



@csrf_exempt
def transcribe_audio(request):
    if request.method == 'POST':
        print('request ', request)
        audio_file = request.FILES.get('audio')
        print('audio_file ', audio_file)
        if audio_file:
            try:
                response =client.audio.transcribe("whisper-1", audio_file)
                transcription = response['text']
                print('transcription ', transcription)
                return JsonResponse({'transcription': transcription})
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)
        else:
            return JsonResponse({'error': 'No audio file provided.'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=400)

