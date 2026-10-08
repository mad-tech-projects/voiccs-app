import os
import sys
import torch
import torchaudio
import numpy as np
import imageio_ffmpeg
import gradio as gr
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Vincular binarios de ffmpeg/ffprobe al PATH de Python
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_exe)
os.environ["PATH"] += os.pathsep + ffmpeg_dir

# Crear enlace simbólico o copia temporal a ffprobe si no existe
ffprobe_path = os.path.join(ffmpeg_dir, "ffprobe")
if not os.path.exists(ffprobe_path):
    try:
        os.symlink(ffmpeg_exe, ffprobe_path)
    except Exception:
        pass

# Carga del modelo Demucs
print("Cargando modelo Demucs...")
model = get_model('htdemucs')
model.eval()
print("Modelo listo.")

def separar_audio(audio_path):
    if not audio_path:
        return None, None
    
    try:
        print(f"Cargando archivo: {audio_path}")
        wav, sr = torchaudio.load(audio_path)
        
        # Asegurar 2 canales (estéreo)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        
        print("Separando pistas con IA...")
        with torch.no_grad():
            sources = apply_model(model, wav.unsqueeze(0), split=True)[0]
        
        pista_instrumental = (sources[0] + sources[1] + sources[2]).cpu()
        pista_voz = sources[3].cpu()
        
        # Formato NumPy para reproducción en Gradio: (muestras, canales)
        inst_numpy = pista_instrumental.numpy().T
        voz_numpy = pista_voz.numpy().T
        
        print("Procesamiento finalizado exitosamente.")
        return (sr, voz_numpy), (sr, inst_numpy)
        
    except Exception as e:
        print(f"Error procesando audio: {e}")
        return None, None

custom_css = """
body, .gradio-container { background: #0B0C10 !important; color: #FFFFFF !important; }
.panel-premium { background: rgba(255, 255, 255, 0.04) !important; border-radius: 20px !important; padding: 20px !important; }
.btn-voiccs { background: linear-gradient(135deg, #7928CA 0%, #FF0080 100%) !important; color: #FFFFFF !important; font-weight: bold !important; border-radius: 12px !important; }
"""

with gr.Blocks(css=custom_css, title="VOICCS AI") as app:
    gr.Markdown("# 🎙️ **VOICCS AI**\n### Tu Estudio Privado de Aislamiento Vocal")
    with gr.Row(elem_classes=["panel-premium"]):
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="🎵 Sube tu canción (MP3, WAV)")
            btn = gr.Button("⚡ SEPARAR PISTAS AHORA", elem_classes=["btn-voiccs"])
        with gr.Column():
            audio_vocal = gr.Audio(label="🎤 Voz Aislada")
            audio_inst = gr.Audio(label="🎸 Pista Instrumental")
            
    btn.click(separar_audio, inputs=[audio_input], outputs=[audio_vocal, audio_inst])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
