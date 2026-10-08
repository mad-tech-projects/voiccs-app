import os
import torchaudio
import torch
import gradio as gr
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Cargar modelo en memoria solo cuando se solicite (Lazy Loading)
model = None

def obtener_modelo():
    global model
    if model is None:
        model = get_model('htdemucs')
        model.eval()
    return model

def separar_audio(audio_path):
    if not audio_path:
        return None, None
    
    # Cargar audio
    wav, sr = torchaudio.load(audio_path)
    
    # Inferencia con IA
    demucs_model = obtener_modelo()
    with torch.no_grad():
        sources = apply_model(demucs_model, wav.unsqueeze(0), split=True)[0]
    
    pista_instrumental = sources[0] + sources[1] + sources[2]
    pista_voz = sources[3]
    
    out_dir = "output_audio"
    os.makedirs(out_dir, exist_ok=True)
    
    voz_path = os.path.join(out_dir, "VOICCS_Voz.wav")
    inst_path = os.path.join(out_dir, "VOICCS_Instrumental.wav")
    
    torchaudio.save(voz_path, pista_voz.cpu(), sr)
    torchaudio.save(inst_path, pista_instrumental.cpu(), sr)
    
    return voz_path, inst_path

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
            btn = gr.Button("⚡ SEPARAR PISTAS", elem_classes=["btn-voiccs"])
        with gr.Column():
            audio_vocal = gr.Audio(label="🎤 Voz Aislada", format="wav")
            audio_inst = gr.Audio(label="🎸 Pista Instrumental", format="wav")
            
    btn.click(separar_audio, inputs=[audio_input], outputs=[audio_vocal, audio_inst])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)
