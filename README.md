🚀 Flux_Img2img: {FROM V3D (video 3D) → to VREAL (video Reale)}

Generazione e modifica di immagini con FLUX, IP-Adapter e Stable Diffusion (SD 1.5, SDXL e FLUX), incluso video da frame interpolati

🎨 Divertiti con Img2Img, FLUX e IP-Adapter per creare immagini straordinarie!

🌟 Introduzione
Questo repository ti permette di generare e modificare immagini con grande libertà creativa, grazie all’integrazione delle tecnologie più avanzate di generazione:

✅ Stable Diffusion 1.5, SDXL, FLUX
✅ FLUX Img2Img – modifica immagini da input esistenti
✅ FLUX Fill – ricostruzione e completamento intelligente di parti mancanti
✅ IP-Adapter – personalizzazione dello stile con immagini di riferimento
✅ Interpolazione frame – creazione di video fluidi a partire da immagini generate

⚙️ Funzionalità principali
🎨 Generazione di immagini da uno sketch o da una foto

👗 Modifica dei vestiti e dettagli con Img2Img

🖼️ Personalizzazione avanzata con IP-Adapter e immagini di riferimento

🧠 FLUX Fill per riempire automaticamente parti mancanti

🎥 Interpolazione di immagini in sequenze animate per creare video fluidi

📦 Contenuto del repository
Il repository include tutto il necessario per usare:

🧠 Stable Diffusion 1.5, SDXL, FLUX

🖌️ FLUX Img2Img + FLUX Fill

🧩 IP-Adapter

🎞️ Interpolazione video da frame

⚠️ Nota importante:
Alcune directory sono state zippate per semplificare il caricamento.
Estrai tutti i file .zip nella cartella principale del progetto prima di avviare qualsiasi script.

🧪 Installazione
bash
Copia
Modifica
# Clona il repository
git clone https://github.com/asprho-arkimete/Flux_Img2img.git
cd Flux_Img2img

# Crea un ambiente virtuale (consigliato: Python 3.10)
python3.10 -m venv vimg2img
source vimg2img/bin/activate   # Su Windows: vimg2img\Scripts\activate.bat

# Installa PyTorch compatibile con la tua GPU
# Per GPU NVIDIA serie 50xx
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Per GPU NVIDIA serie 30xx e 40xx
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Installa i requisiti generali
pip install -r requisiti.txt

# Avvia l'interfaccia
python img2img.py
📥 Download modelli necessari
🔄 Interpolazione frame
Scarica da: frame-interpolation-pytorch

➤ Copia le cartelle models e pretrained_models nella directory del progetto

🤖 Modelli HuggingFace
I modelli vengono scaricati automaticamente nella cartella:
C:\hf_download\hub

➤ Se vuoi cambiare directory, modifica demo_gradio.py:

python
Copia
Modifica
os.environ['HF_HOME'] = r'C:\hf_download'
🧠 Modelli e LoRA
➤ Scaricabili da Civitai

Compatibili con Stable Diffusion 1.5 e FLUX D1, S1

💇 Hair Segmentation
➤ Scarica da: pytorch-hair-segmentation

➤ Copia i modelli nella cartella:
pytorch-hair-segmentation/models

📄 Consulta anche il file installazione.txt per ulteriori dettagli.

🔗 Repository GitHub
➡️ https://github.com/asprho-arkimete/Flux_Img2img
