import os
import torchaudio
import torch
import gradio as gr
from demucs.pretrained import get_model
from demucs.apply import apply_model

# 1. Cargar modelo de IA de separación de audio (HT-Demucs de Meta)
model = get_model('htdemucs')
model.eval()

def separar_audio(audio_path):
    if not audio_path:
        return None, None
    
    # Cargar audio enviado por el usuario
    wav, sr = torchaudio.load(audio_path)
    
    # Separación con Inteligencia Artificial
    with torch.no_grad():
        sources = apply_model(model, wav.unsqueeze(0), split=True)[0]
    
    # Fuentes: 0=Batería, 1=Bajo, 2=Otros, 3=Voz
    pista_instrumental = sources[0] + sources[1] + sources[2]
    pista_voz = sources[3]
    
    out_dir = "output_audio"
    os.makedirs(out_dir, exist_ok=True)
    
    voz_path = os.path.join(out_dir, "VOICCS_Voz.wav")
    inst_path = os.path.join(out_dir, "VOICCS_Instrumental.wav")
    
    torchaudio.save(voz_path, pista_voz.cpu(), sr)
    torchaudio.save(inst_path, pista_instrumental.cpu(), sr)
    
    return voz_path, inst_path

# 2. Diseño de la Interfaz Estilo Dark Mode & Neon Premium
custom_css = """
body, .gradio-container {
    background: #0B0C10 !important;
    color: #FFFFFF !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
.panel-premium {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5) !important;
}
.btn-voiccs {
    background: linear-gradient(135deg, #7928CA 0%, #FF0080 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: bold !important;
    font-size: 16px !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
}
.btn-voiccs:hover {
    transform: scale(1.02);
    box-shadow: 0 0 20px rgba(255, 0, 128, 0.4) !important;
}
"""

with gr.Blocks(css=custom_css, title="VOICCS AI") as app:
    gr.Markdown(
        """
        # 🎙️ **VOICCS AI**
        ### Tu Estudio Privado de Aislamiento Vocal con Inteligencia Artificial
        ---
        """
    )
    
    with gr.Row(elem_classes=["panel-premium"]):
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="🎵 Sube tu canción o audio (MP3, WAV, M4A)")
            btn = gr.Button("⚡ SEPARAR PISTAS AHORA", elem_classes=["btn-voiccs"])
        
        with gr.Column(scale=1):
            audio_vocal = gr.Audio(label="🎤 Voz Aislada", format="wav")
            audio_inst = gr.Audio(label="🎸 Pista Instrumental", format="wav")
            
    btn.click(
        fn=separar_audio, 
        inputs=[audio_input], 
        outputs=[audio_vocal, audio_inst]
    )

app.launch()
