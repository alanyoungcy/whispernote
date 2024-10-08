# transcriber/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.gradio_interface, name='gradio_interface'),
    path('transcribe/', views.transcribe_audio, name='transcribe_audio'),

]
