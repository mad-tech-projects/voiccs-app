import os
import torch
import torchaudio
import gradio as gr
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Cargar el modelo de Demucs
model = get_model('htdemucs')
model.eval()

def separar_audio(audio_path):
    if not audio_path:
        return None, None
    
    try:
        # Cargar archivo de audio
        wav, sr = torchaudio.load(audio_path)
        
        # Asegurar que sea stereo de 2 canales
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        
        # Procesar con la IA
        with torch.no_grad():
            sources = apply_model(model, wav.unsqueeze(0), split=True)[0]
        
        # Pista instrumental (Bajo + Batería + Otros) y Voz
        pista_instrumental = sources[0] + sources[1] + sources[2]
        pista_voz = sources[3]
        
        # Carpeta de salida
        out_dir = os.path.abspath("output_audio")
        os.makedirs(out_dir, exist_ok=True)
        
        voz_path = os.path.join(out_dir, "VOICCS_Voz.wav")
        inst_path = os.path.join(out_dir, "VOICCS_Instrumental.wav")
        
        # Guardar archivos WAV
        torchaudio.save(voz_path, pista_voz.cpu(), sr)
        torchaudio.save(inst_path, pista_instrumental.cpu(), sr)
        
        return voz_path, inst_path
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
            audio_input = gr.Audio(type="filepath", label="🎵 Sube tu canción (MP3, WAV, M4A)")
            btn = gr.Button("⚡ SEPARAR PISTAS AHORA", elem_classes=["btn-voiccs"])
        with gr.Column():
            audio_vocal = gr.Audio(label="🎤 Voz Aislada")
            audio_inst = gr.Audio(label="🎸 Pista Instrumental")
            
    btn.click(separar_audio, inputs=[audio_input], outputs=[audio_vocal, audio_inst])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
