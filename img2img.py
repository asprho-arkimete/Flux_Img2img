# Standard libraries
import os
import re
import gc
import time
import shutil
import threading

# GUI
import tkinter as tk
from tkinter import filedialog, ttk

# Image/video processing
import cv2
import numpy as np
from PIL import Image, ImageTk
import imageio
from moviepy import VideoFileClip

# AI and Diffusers
import torch
from transformers import CLIPVisionModel
from diffusers import (
    AutoencoderKLWan,
    WanVideoToVideoPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image, load_video, export_to_video

# Pose and edge detectors
from controlnet_aux import CannyDetector, OpenposeDetector

# Translation
from deep_translator import GoogleTranslator

# Hugging Face Hub
from huggingface_hub import login, HfApi, hf_hub_download

from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image

import traceback
import os, gc, re
from PIL import Image
import torch
from deep_translator import GoogleTranslator
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler,StableDiffusionControlNetPipeline
from controlnet_aux import OpenposeDetector

from tkinterdnd2 import DND_FILES
from tkinterdnd2 import TkinterDnD

from diffusers.utils import export_to_video

import torch
from diffusers import (
    DDIMScheduler,
    MotionAdapter,
    PIAPipeline,
)
from diffusers.utils import export_to_gif, load_image




# --- CONFIGURAZIONI ---
output_dirs = ["modelli", "Lora", "frames", "output"]

# --- LOGIN HUGGINGFACE ---
filepath=None

# --- SETUP TORCH ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- VARIABILI GLOBALI ---
framescollection = []
maskcollecollection=[]
current_frame_index = 0
img_canvas = None

# --- GUI SETUP ---
window = TkinterDnD.Tk()
window.title("Img2img")
window.geometry("1500x1000")  # Ho allargato un po' per dare spazio
window.resizable(False, False)

# Frame principale orizzontale
frame_superiore = tk.Frame(window)
frame_superiore.grid(row=0, column=0, sticky="nw", padx=10, pady=10)

status_var = tk.StringVar(value="Pronto")
status_bar = tk.Label(window, textvariable=status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.grid(row=3, column=0, columnspan=2, sticky='ew')

# Canvas a sinistra
frame_canvas = tk.Frame(frame_superiore)
frame_canvas.pack(side='left', padx=10)

frame = tk.Canvas(frame_canvas, width=512, height=512, bg='red')
frame.pack()

# Bottoni a destra
frame_strumenti = tk.Frame(frame_superiore)
frame_strumenti.pack(side='left', padx=10)



framestrumenti2 = tk.Frame(window)
framestrumenti2.grid(row=1, column=0)

frame_prompt = tk.Frame(window)
frame_prompt.grid(row=2, column=0, columnspan=2, sticky='ew', pady=10)

def cambia_frame(valore):
    global current_frame_index
    index = int(float(valore)) - 1
    if 0 <= index < len(framescollection):
        current_frame_index = index
        mostra_immagine(framescollection[index])
        status_var.set(f"Frame {index+1}/{len(framescollection)}")

scorri_frames = tk.Scale(framestrumenti2, from_=1, to=100, orient=tk.HORIZONTAL, command=cambia_frame)
scorri_frames.grid(row=0, column=0, columnspan=2, pady=10)

prompt = tk.Text(frame_prompt, width=50, height=10)
prompt.grid(row=1, column=0, padx=5, pady=5)
prompt.insert("1.0", "inserisci prompt positivo")
tk.Label(frame_prompt, text="Prompt Positivo").grid(row=0, column=0, sticky='w')

negative = tk.Text(frame_prompt, width=50, height=10)
negative.grid(row=1, column=1, padx=5, pady=5)
negative.insert("1.0", (
    "worst quality, low quality:1.4, illustration, 3d, 2d, painting, cartoons, sketch, blur, "
    "blurry, grainy, low resolution, aliasing, dithering, distorted, jpeg artifacts, "
    "compression artifacts, overexposed, high contrast, bad anatomy, watermark, text, error"
))
tk.Label(frame_prompt, text="Prompt Negativo").grid(row=0, column=1, sticky='w')

def f_preset_inpaint(event=None):
    video_preset = ['No_Preset'] + [
        preset for preset in os.listdir("preset_inpainting")
        if '_mask' not in preset and preset.endswith(('.mp4', '.avi', '.mov'))
    ]
    preset_inpaint.config(values=video_preset)


def carica_mask_preset():
    global collectionmaskD, collectionmaskS, maskcollecollection
    # Inizializzazione
    collectionmaskD = []
    collectionmaskS = []
    
    if preset_inpaint.get() == "No_Preset":
        print("Nessun preset selezionato")
        return
    
    print(f"Caricamento preset: {preset_inpaint.get()}")
    
    # Carico i file
    base_dir = os.path.join("preset_inpainting", preset_inpaint.get().split('.')[0])
    filemaskD = f"{base_dir}_maskD.mp4"
    filemaskS = f"{base_dir}_maskS.mp4"
    
    print(f"Percorso maskD: {filemaskD}")
    print(f"Percorso maskS: {filemaskS}")
    print(f"File maskD esiste: {os.path.exists(filemaskD)}")
    print(f"File maskS esiste: {os.path.exists(filemaskS)}")
    
    # Funzioni per estrarre i frame in thread separati
    def extract_frames_D():
        global collectionmaskD
        try:
            clip = VideoFileClip(filemaskD)
            collectionmaskD = [frame for frame in tqdm(clip.iter_frames())]
            print(f"Estratti {len(collectionmaskD)} frame da maskD")
            clip.close()
        except Exception as e:
            print(f"Errore nell'estrazione dei frame da maskD: {e}")
            collectionmaskD = []
    
    def extract_frames_S():
        global collectionmaskS
        try:
            clip = VideoFileClip(filemaskS)
            collectionmaskS = [frame for frame in tqdm(clip.iter_frames())]
            print(f"Estratti {len(collectionmaskS)} frame da maskS")
            clip.close()
        except Exception as e:
            print(f"Errore nell'estrazione dei frame da maskS: {e}")
            collectionmaskS = []
    
    # Avvio thread per estrazione frame
    threads = []
    if os.path.exists(filemaskD):
        thread_D = threading.Thread(target=extract_frames_D)
        thread_D.start()
        threads.append(thread_D)
    
    if os.path.exists(filemaskS):
        thread_S = threading.Thread(target=extract_frames_S)
        thread_S.start()
        threads.append(thread_S)
    
    # Attendi che entrambi i thread completino
    for thread in threads:
        thread.join()
    
    if not collectionmaskD and not collectionmaskS:
        print("Nessun frame estratto dalle maschere")
        return
    
    # Apro il video per ottenere dimensioni e fps
    video_ref = None
    try:
        if os.path.exists(filemaskD):
            video_ref = VideoFileClip(filemaskD)
        elif os.path.exists(filemaskS):
            video_ref = VideoFileClip(filemaskS)
        else:
            print("Nessun file video di riferimento trovato")
            return
        
        # Creo il fondo con le dimensioni corrette
        width, height = int(video_ref.w), int(video_ref.h)
        print(f"Dimensioni video: {width}x{height}")
        
        # Unisco i frame
        collectionmasknew = []
        
        # Assicurati che ci siano frame da elaborare
        max_frames = max(len(collectionmaskD) if collectionmaskD else 0, 
                         len(collectionmaskS) if collectionmaskS else 0)
        
        print(f"Elaborazione di {max_frames} frame")
        
        for i in range(max_frames):
            # Crea una nuova immagine per ogni frame
            sfondo = Image.new("RGB", (width, height), (0, 0, 0))
            
            # Aggiungi maskD se disponibile per questo frame
            if collectionmaskD and i < len(collectionmaskD):
                try:
                    maskD_img = Image.fromarray(collectionmaskD[i].astype('uint8'))
                    maskD_alpha = maskD_img.convert("L")  # Usa l'immagine stessa come mask
                    sfondo.paste(maskD_img, (0, 0), maskD_alpha)
                except Exception as e:
                    print(f"Errore durante l'elaborazione del frame {i} di maskD: {e}")
            
            # Aggiungi maskS se disponibile per questo frame
            if collectionmaskS and i < len(collectionmaskS):
                try:
                    maskS_img = Image.fromarray(collectionmaskS[i].astype('uint8'))
                    maskS_alpha = maskS_img.convert("L")  # Usa l'immagine stessa come mask
                    sfondo.paste(maskS_img, (0, 0), maskS_alpha)
                except Exception as e:
                    print(f"Errore durante l'elaborazione del frame {i} di maskS: {e}")
            
            collectionmasknew.append(np.array(sfondo))
        
        maskcollecollection = collectionmasknew
        print(f"Creati {len(collectionmasknew)} frame combinati")
        
        # Salvo il file
        output_path = './preset_inpainting//mask_preset.mp4'
        
        print(f"Salvataggio del video in: {output_path}")
        
        videonew = ImageSequenceClip(collectionmasknew, fps=video_ref.fps)
        videonew.write_videofile(output_path, codec='libx264')
        print("Video salvato con successo")
    
    except Exception as e:
        print(f"Errore durante l'elaborazione delle maschere: {e}")
    finally:
        # Chiudi i video
        if video_ref:
            video_ref.close()

def aprifile_preset(event=None):
    global filepath, scorri_frames
    if preset_inpaint.get() == "No_Preset":
        print("Nessun preset selezionato")
        return
    
    print(f"Apertura preset: {preset_inpaint.get()}")
    
    try:
        carica_mask_preset()
        print("Maschere caricate con successo")
        filepath = os.path.join("preset_inpainting", preset_inpaint.get())
        if not filepath:
            return
        threading.Thread(target=extract_frames_thread, args=(filepath,), daemon=True).start()
    
    except Exception as e:
        print(f"Errore durante il caricamento delle maschere: {e}")


   
# Combobox
preset_inpaint = ttk.Combobox(frame_prompt)
preset_inpaint.grid(row=1, column=2, padx=10)

# Binding e inizializzazione
preset_inpaint.bind('<Button-1>', f_preset_inpaint)
preset_inpaint.bind('<<ComboboxSelected>>',aprifile_preset)
f_preset_inpaint()


def load_modelli():
    global Modelli
    try:
        if os.path.exists('modelli'):
            models = [os.path.basename(model) for model in os.listdir('modelli')]
            Modelli.config(values=models)
        else:
            Modelli.config(values=["Directory 'modelli' non trovata"])
    except Exception as e:
        print(f"Errore durante il caricamento dei modelli: {e}")

Modelli = ttk.Combobox(framestrumenti2)
Modelli.grid(row=2, column=0)
# Correzione dell'evento: usa <<ComboboxSelected>> invece di <<Botton-1>>
Modelli.bind('<<ComboboxSelected>>', lambda event: load_modelli())
# Oppure se volevi un click del mouse usa <Button-1>
# Modelli.bind('<Button-1>', lambda event: load_modelli())
tk.Label(framestrumenti2, text="Modelli").grid(row=1, column=0)
load_modelli()

Lora = ttk.Combobox(framestrumenti2)
Lora.grid(row=2, column=1)
tk.Label(framestrumenti2, text="Lora").grid(row=1, column=1)
Lora.bind('<Button-1>', lambda event: caricaLora())

Steps = tk.Scale(framestrumenti2, from_=1, to=100, orient=tk.HORIZONTAL)
Steps.grid(row=2, column=2, padx=10)
Steps.set(25)
tk.Label(framestrumenti2, text="Steps").grid(row=1, column=2)

Cfg = tk.Scale(framestrumenti2, from_=1.0, to=20.0, resolution=0.1, orient=tk.HORIZONTAL)
Cfg.grid(row=2, column=3, padx=10)
Cfg.set(10.0)
tk.Label(framestrumenti2, text="CFG").grid(row=1, column=3)

modifica= tk.Scale(framestrumenti2, from_=0.01, to=1.00, resolution=0.01, orient=tk.HORIZONTAL)
modifica.grid(row=2, column=4, padx=10)
modifica.set(0.50)
tk.Label(framestrumenti2, text="Modifica").grid(row=1, column=4)

lorascale = tk.Scale(framestrumenti2, from_=0.01, to=1.00, resolution=0.01, orient=tk.HORIZONTAL)
lorascale.grid(row=2, column=5, padx=10)
lorascale.set(1.0)
tk.Label(framestrumenti2, text="Lora scale").grid(row=1, column=5)

exadi = tk.Scale(framestrumenti2, from_=2, to=20, resolution=1, orient=tk.HORIZONTAL)
exadi.grid(row=2, column=6, padx=10)
exadi.set(1.0)

tk.Label(framestrumenti2, text="Espandi mask").grid(row=1, column=6)

risoluzione = ttk.Combobox(framestrumenti2, values=[
    # Formati quadrati e verticali
    '512,512', '512,768', '512,960', '512,1024', '640,960',
    '768,768', '768,1024', '768,1280', '1024,1024', '1024,1280',
    '1024,1536', '1152,1536', '1280,1280', '1280,1536',
    '1280,1792', '1536,1536', '1536,2048', '1792,2048',
    '2048,2048', '2048,2560', '2560,2560', '2560,3072',
    '3072,3072', '3072,3584', '3584,3584', '3584,4000',
    '4000,4000',
    
    # Formati 16:9 aggiunti
    '640,360',    # 360p
    '854,480',    # 480p
    '1280,720',   # 720p
    '1600,900',   # HD+
    '1920,1080',  # Full HD
    '2560,1440',  # 2K QHD
    '3200,1800',  # Retina
    '3840,2160',  # 4K UHD
])

risoluzione.grid(row=1, column=7, padx=10)
risoluzione.set('1280,720')


import webbrowser
import winreg
def F_Avvio_Daz3d():
    desktop = os.path.expandvars(r"%USERPROFILE%\Desktop")
    possible_links = [
        "DAZ Studio 4.23 (64-bit).lnk",
        "DAZ Studio 4.22 (64-bit).lnk",
        "DAZ Studio.lnk",
        "Daz3D.lnk",
    ]

    for name in possible_links:
        full_path = os.path.join(desktop, name)
        if os.path.exists(full_path):
            print(f"Avvio DAZ3D tramite: {full_path}")
            try:
                os.startfile(full_path)
                return
            except Exception as error:
                print("Errore nell'avvio del collegamento.")
                print(f"Errore: {error}")
    
    print("DAZ 3D non trovato. Apro il sito per scaricarlo...")
    print("IMPORTANTE: CREA un collegamento sul DESKTOP")
    webbrowser.open("https://www.daz3d.com/")

Avvio_daz3d = tk.Button(framestrumenti2, text="Avvio Daz3d", bg="skyblue", command=F_Avvio_Daz3d)
Avvio_daz3d.grid(row=1, column=8, padx=10)


# --- FUNZIONI ---
def mostra_immagine(img_path):
    global img_canvas
    try:
        img = Image.open(img_path)
        w, h = img.size
        if w >= h:
            img = img.resize((512, (512 * h) // w), Image.LANCZOS)
        else:
            img = img.resize(((512 * w) // h, 512), Image.LANCZOS)

        img_canvas = ImageTk.PhotoImage(img)
        frame.delete("all")
        frame.create_image(256, 256, anchor="center", image=img_canvas)
    except Exception as e:
        status_var.set(f"Errore immagine: {e}")

import os
from PIL import Image
from decord import VideoReader, cpu, gpu
from tqdm import tqdm
import torch
import subprocess
import subprocess
from PIL import Image
from tqdm import tqdm
import os
import glob

def extract_frames_thread(filepath):
    global framescollection, scorri_frames

    try:
        status_var.set("🎞️ Estrazione frames in corso...")
        framescollection.clear()
        os.makedirs("frames", exist_ok=True)

        print(f"⚙️ Estrazione con FFmpeg + CUDA")

        # Comando FFmpeg per estrarre tutti i frame con accelerazione GPU
        output_pattern = os.path.join("frames", "frame-%04d.png")
        cmd = [
            "ffmpeg",
            "-hwaccel", "cuda", 
            "-i", filepath,
            output_pattern,
            "-y"
        ]

        subprocess.run(cmd, check=True)

        # Carica tutti i frame senza ridimensionarli
        frame_files = sorted(glob.glob("frames/frame-*.png"))
        
        for frame_path in tqdm(frame_files, desc="Caricamento frame"):
            framescollection.append(frame_path)

        if framescollection:
            scorri_frames.config(from_=1, to=len(framescollection))
            mostra_immagine(framescollection[0])
            status_var.set(f"✅ Estratti {len(framescollection)} frame")
        else:
            status_var.set("⚠️ Nessun frame estratto")

    except Exception as e:
        import traceback
        traceback.print_exc()
        status_var.set(f"❌ Errore: {str(e)}")

def aprifile():
    global filepath,scorri_frames
    filepath = filedialog.askopenfilename(filetypes=[("Video o Immagini", "*.mp4 *.avi *.mov *.jpg *.png *.jpeg")])
    if not filepath:
        return
    if filepath.lower().endswith((".jpg", ".jpeg", ".png")):
        # remove FRames
        for oldf in tqdm(os.listdir('./frames'), desc= "Pulizia frames precedenti"):
            os.remove(os.path.join('./frames',oldf))
        folder = os.path.dirname(filepath)
        framescollection[:] = [
            os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if framescollection:
            scorri_frames.config(to=len(framescollection))
            mostra_immagine(framescollection[0])
            status_var.set(f"Caricate {len(framescollection)} immagini")
        else:
            status_var.set("Nessuna immagine trovata")
    else:
        for oldf in tqdm(os.listdir('./frames'), desc= "Pulizia frames precedenti"):
            os.remove(os.path.join('./frames',oldf))
        threading.Thread(target=extract_frames_thread, args=(filepath,), daemon=True).start()

 
def caricaLora():
    try:
        if os.path.exists('Lora'):
            loras = ['No_Lora'] + [os.path.splitext(l)[0] for l in os.listdir('Lora') if l.lower().endswith(('.safetensors', '.ckpt', '.pt'))]
            Lora['values'] = loras
            status_var.set(f"Trovati {len(loras)-1} LoRA")
        else:
            os.makedirs("Lora", exist_ok=True)
    except Exception as e:
        status_var.set(f"Errore LoRA: {e}")
 

generator=None
import torch, os, gc, traceback, re
import cv2
from PIL import Image
import numpy as np
from deep_translator import GoogleTranslator
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler, DDIMScheduler
from controlnet_aux import OpenposeDetector
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPTokenizer, CLIPTextModel
# Ecco una versione modificata della funzione Img2Img() con le correzioni necessarie
def Img2Img():
    global Cfg, Steps, framescollection, scorri_frames, prompt, Lora, lorascale, modifica, scale
    global Modelli, negative, ip_adapter_var, ip_image_path1, ip_image_path2,risoluzione,controlNetSelect
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    print(f"🖥️ Dispositivo selezionato: {device}")

    model_path = f"modelli/{Modelli.get()}"
    model = f"{model_path}.safetensors" if os.path.exists(f"{model_path}.safetensors") else model_path
    if not os.path.exists(model):
        print(f"⚠️ Modello non trovato: {model}")
        return

    is_lcm = "lcm" in model.lower()
    is_pony = "pony" in model.lower()

    prompt_text_raw = prompt.get('1.0', 'end').strip()
    if not prompt_text_raw or prompt_text_raw == 'inserisci prompt positivo':
        print("⚠️ Prompt non valido.")
        return

    neg_prompt = negative.get('1.0', 'end').strip()
    neg_prompt = "" if neg_prompt == 'inserisci prompt negativo' else neg_prompt

    def parse_multiprompt_ranges(text, total_frames):
            prompts = text.split(';')
            parsed = []
            for pr in prompts:
                pr = pr.strip()
                if ':' in pr:
                    try:
                        index, txt = pr.split(':', 1)
                        index = int(index.strip())
                        loras = re.findall(r'<(.*?)>', txt)
                        parsed.append((index, txt.strip(), loras))
                    except ValueError:
                        print(f"⚠️ Errore parsing indice in '{pr}'")
            parsed.sort()
            result = [None] * total_frames
            for i, (start_idx, txt, loras) in enumerate(parsed):
                end_idx = parsed[i+1][0] if i+1 < len(parsed) else total_frames
                for j in range(start_idx, min(end_idx, total_frames)):
                    result[j] = {'image': j, 'prompt': txt, 'loras': loras}
            return result

    try:
        if ';' in prompt_text_raw and ':' in prompt_text_raw:
            prompt_list = parse_multiprompt_ranges(prompt_text_raw, len(framescollection))
        else:
            current_frame = int(scorri_frames.get()) - 1 if scorri_frames.get() else 0
            prompt_list = [{
                'image': current_frame,
                'prompt': prompt_text_raw.rstrip(';'),
                'loras': [Lora.get()] if Lora and Lora.get() not in ('', 'No_Lora') else []
            }]
    except Exception as e:
        print(f"❌ Errore parsing prompt: {e}")
        return

    try:
        translator = GoogleTranslator(source='it', target='en')
        for item in prompt_list:
            try:
                item['translated'] = translator.translate(item['prompt'])
            except:
                item['translated'] = item['prompt']
    except:
        for item in prompt_list:
            item['translated'] = item['prompt']

    try:
        # Definisci il tipo di dati in base al dispositivo
        dtype = torch.float16
        controlnet=None
        #'open pose', 'face_tongue'
        if controlNetSelect=='open pose':
            # Carica il modello ControlNet con il tipo di dati specificato
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose", 
                torch_dtype=dtype, 
                use_safetensors=True
            )
        
            processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        else:
            controlnet = ControlNetModel.from_pretrained("CrucibleAI/ControlNetMediaPipeFace", subfolder="diffusion_sd15",torch_dtype=dtype)

        # Carica il modello principale
        try:
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                model, 
                controlnet=controlnet, 
                torch_dtype=dtype
            )
        except:
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                model, 
                controlnet=controlnet, 
                torch_dtype=dtype
            )

        # Imposta lo scheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
       
        # --- IMPORTANTE: Gestione corretta del dispositivo ---
        if device == "cuda":
            # Sposta tutto il pipeline sul dispositivo CUDA
            pipe = pipe.to(device)
            # Abilita l'offload solo dopo aver spostato tutto su CUDA
            pipe.enable_model_cpu_offload()
            pipe.enable_xformers_memory_efficient_attention = False
        else:
            pipe = pipe.to(device)

    except Exception as e:
        print(f"❌ Errore nel caricamento modelli: {e}")
        traceback.print_exc()
        return

    # --- Logica di controllo frame ---
    is_multi_prompt = ';' in prompt_text_raw and ':' in prompt_text_raw
    try:
        scale_value = int(scale.get()) if scale.get() else 1
    except:
        scale_value = 1

    # IP Adapter - Gestione adattamento multiplo
    ip_adapter_enabled = False  # Flag per tracciare se IP Adapter è attivo
    ip_images = []  # Lista vuota per le immagini IP Adapter
    weight_names = []  # Lista per i nomi dei pesi degli adapter
    adapter_scales = []  # Lista per le scale degli adapter

    # Verifica dei percorsi IP Adapter
    face_ip_path = ip_image_path1 if 'ip_image_path1' in globals() else None
    image_ip_path = ip_image_path2 if 'ip_image_path2' in globals() else None
    use_face_adapter = face_ip_path and os.path.exists(face_ip_path)
    use_image_adapter = image_ip_path and os.path.exists(image_ip_path)
    
    # Tentativo di caricare IP Adapter solo se necessario
    if use_face_adapter or use_image_adapter:
        try:
            # Cambia il scheduler a DDIM (richiesto per IP-Adapter)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            
            # Scale per IP adapter - Supporta valori separati per ciascun adapter
            ip_scale_values = ip_adapter_var.get()
            ip_scale_image = float(ip_scale_values)
            ip_scale_face = float(ip_scale_values)
            
            # Configura caricamento di uno o entrambi gli adapter
            model_path = "h94/IP-Adapter"
            
            # Carica encoder condiviso (opzionale, per ottimizzazione)
            try:
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    model_path,
                    subfolder="models/image_encoder",
                    torch_dtype=dtype
                )
                print("✅ Encoder immagini caricato per IP Adapter")
            except Exception as e:
                print(f"⚠️ Non è stato possibile caricare l'encoder specifico: {e}")
                image_encoder = None
            
            # Prepara le liste per immagini e pesi adapter
            if use_image_adapter and use_face_adapter:
                # Entrambi gli adapter
                print("🖼️👤 Utilizzo entrambi gli adapter: immagine completa e volto")
                weight_names = ["ip-adapter_sd15.bin", "ip-adapter-full-face_sd15.bin"]
                
                # Carica le immagini
                full_ip_image = Image.open(image_ip_path).convert("RGB").resize((768, 768), Image.LANCZOS)
                face_ip_image = Image.open(face_ip_path).convert("RGB").resize((768, 768), Image.LANCZOS)
                
                # Assegna le immagini e le scale
                ip_images = [full_ip_image, face_ip_image]
                adapter_scales = [ip_scale_image, ip_scale_face]
                
                # Carica gli adapter
                pipe.load_ip_adapter(
                    model_path,
                    subfolder="models",
                    weight_name=weight_names,
                    image_encoder=image_encoder
                )
                
                print(f"✅ IP Adapter multipli caricati con scale: {adapter_scales}")
                ip_adapter_enabled = True
                
            elif use_image_adapter:
                print("🖼️ Uso immagine completa come riferimento di adapter")
                print(f"📁 Percorso immagine IP Full: {image_ip_path}")
                
                # Carica l'adapter
                pipe.load_ip_adapter(
                    model_path,
                    subfolder="models",
                    weight_name="ip-adapter_sd15.bin",
                    image_encoder=image_encoder,
                    torch_dtype=dtype
                )
                
                # Carica e prepara l'immagine
                full_ip_image = Image.open(image_ip_path).convert("RGB").resize((768, 768), Image.LANCZOS)
                ip_images = [full_ip_image]
                adapter_scales = [ip_scale_image]
                
                ip_adapter_enabled = True
                print(f"✅ IP Adapter Immagine Intera caricato con scala: {ip_scale_image}")
                
            elif use_face_adapter:
                print("👤 Uso immagine del viso come riferimento di adapter")
                print(f"📁 Percorso immagine IP Faccia: {face_ip_path}")
                
                # Carica l'adapter
                pipe.load_ip_adapter(
                    model_path,
                    subfolder="models",
                    weight_name="ip-adapter-full-face_sd15.bin",
                    image_encoder=image_encoder,
                    torch_dtype=dtype
                )
                
                # Carica e prepara l'immagine
                face_ip_image = Image.open(face_ip_path).convert("RGB").resize((768, 768), Image.LANCZOS)
                ip_images = [face_ip_image]
                adapter_scales = [ip_scale_face]
                
                ip_adapter_enabled = True
                print(f"✅ IP Adapter Faccia caricato con scala: {ip_scale_face}")
            
            # Imposta le scale per gli adapter se ci sono più adapter
            if len(adapter_scales) > 0:
                pipe.set_ip_adapter_scale(adapter_scales)
            
            # Assicurati che la pipeline sia sul dispositivo corretto dopo le modifiche
            pipe = pipe.to(device)
            
        except Exception as e:
            print(f"❌ Errore IP Adapter: {e}")
            traceback.print_exc()
            ip_adapter_enabled = False
            ip_images = []

    for prompt_item in prompt_list:
        frame_index = prompt_item['image']
        if frame_index < 0 or frame_index >= len(framescollection):
            continue

        # --- Condizione di generazione immagine ---
        generate_this_frame = False
        if prompt_text_raw.strip() == ';':
            generate_this_frame = True
        elif is_multi_prompt:
            if frame_index==0 or frame_index % scale_value == 0:
                generate_this_frame = True
        else:
            current_frame = int(scorri_frames.get()) - 1 if scorri_frames.get() else 0
            if frame_index == current_frame:
                generate_this_frame = True

        if not generate_this_frame:
            print(f"⏭️ Skip frame {frame_index}")
            continue

        try:
            current_image = framescollection[frame_index]
            if isinstance(current_image, str) and os.path.exists(current_image):
                current_image = Image.open(current_image).convert("RGB")
            elif not isinstance(current_image, Image.Image):
                continue
            w,h= current_image.size
            if w>= h:
                #w*768: h*X
                X= (768*h)//w
                current_image = current_image.resize((768,X), Image.BICUBIC)
            else:
                #h*768: w*X
                X= (768*w)//h
                current_image = current_image.resize((X,768), Image.BICUBIC)

            
            if controlNetSelect=='open pose':
                pose_image = processor(current_image, hand_and_face=True)
                pose_image = pose_image.resize(current_image.size, Image.BILINEAR)
                pose_image.save(f"./output_image/pose_img_{frame_index}.png")
            else:
                pose_image= current_image
                pose_image.save(f"./output_image/face_tongue_{frame_index}.png")
            

            

            current_prompt = prompt_item['translated']
            if prompt_item['loras'] and lorascale:
                for lora_name in prompt_item['loras']:
                    if lora_name != 'No_Lora':
                        lora_path = f"lora/{lora_name}"
                        if os.path.exists(lora_path):
                            try:
                                pipe.load_lora_weights(lora_path, weight_name=lora_name, weight=float(lorascale.get()))
                                pipe.to(device)
                            except Exception as e:
                                print(f"⚠️ LoRA non caricata: {e}")

            # Parametri per la generazione immagine
            wg,hg= 1024,1024
            # Ottieni la risoluzione selezionata dall'utente (es. "1024,1024")
            res_string = risoluzione.get()
            if not res_string:
                res_string = "1024,1024"  # fallback se non selezionato

            # Converte in interi
            max_w, max_h = map(int, res_string.split(','))

            # Calcolo proporzionale mantenendo aspect ratio
            if w >= h:
                wg = max_w
                hg = (max_w * h) // w
            else:
                hg = max_h
                wg = (max_h * w) // h
            # Tokenizer e text encoder
            tokenizer = CLIPTokenizer.from_pretrained("SG161222/Realistic_Vision_V6.0_B1_noVAE", subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained("SG161222/Realistic_Vision_V6.0_B1_noVAE", subfolder="text_encoder").to("cpu")

            def get_prompt_embeds(prompt: str):
                inputs = tokenizer(prompt, padding="max_length", truncation=False, return_tensors="pt")
                
                input_ids = inputs.input_ids.to(text_encoder.device)
                
                # Se il prompt è più lungo di 77 token, spezza in blocchi da 75 + CLS/SEP token
                if input_ids.shape[1] > 77:
                    chunks = input_ids[0].split(75)
                    embeds = []

                    for chunk in chunks:
                        chunk = chunk.unsqueeze(0)  # batch size = 1
                        padded = torch.nn.functional.pad(chunk, (0, 77 - chunk.shape[1]), value=tokenizer.pad_token_id)
                        emb = text_encoder(padded)[0]
                        embeds.append(emb)

                    prompt_embeds = torch.cat(embeds, dim=1)
                    prompt_embeds = prompt_embeds[:, :77, :]  # taglia se troppo lungo alla fine
                else:
                    prompt_embeds = text_encoder(input_ids)[0]

                return prompt_embeds
            ebbending=get_prompt_embeds(current_prompt)
            generation_params = {
                "prompt_embeds": ebbending,
                "negative_prompt": neg_prompt,
                "image": current_image,
                "control_image": pose_image,
                "num_inference_steps": int(Steps.get()),
                "guidance_scale": float(Cfg.get()),
                "strength": float(modifica.get()),
                "width": wg,
                "height": hg,
                "generator": torch.manual_seed(1000) if prompt_text_raw.strip() == ';' else torch.Generator(device=device).manual_seed(torch.seed())
            }
            
            # Aggiungi l'immagine IP adapter se è disponibile
            if ip_adapter_enabled and ip_images:
                generation_params["ip_adapter_image"] = ip_images

            # Generazione immagine con gestione degli errori
            try:
                result = pipe(**generation_params).images[0]
                result.save(f"./output_image/sd_{frame_index}.png")
                print(f"✅ Immagine salvata: sd_{frame_index}.png")
            except Exception as err:
                print(f"❌ Errore nella generazione dell'immagine: {err}")
                traceback.print_exc()

        except Exception as err:
            print(f"❌ Errore frame {frame_index}: {err}")
            traceback.print_exc()

    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    print("🏁 Generazione completata.")

        

# --- BOTTONI ---
# Creiamo un frame solo per i primi due bottoni affiancati
frame_bottoni_superiori = tk.Frame(frame_strumenti)
frame_bottoni_superiori.pack(pady=5)
# ----- RIGA 1: img2img, text_to_image, Inpainting -----
frame_riga1 = tk.Frame(frame_bottoni_superiori)
frame_riga1.pack()

# Bottone img2img
tk.Button(frame_riga1, text="img2img", width=25, bg='orange', fg='black',
          activebackground='#FF8C00', command=Img2Img).pack(side='left', padx=0)


def text_to_Image():
    global Cfg, Steps, framescollection, scorri_frames, prompt, Lora, lorascale, scale
    global Modelli, negative, ip_adapter_var, ip_image_path1, ip_image_path2
    import torch, os, gc, traceback, re
    import cv2
    from PIL import Image
    import numpy as np
    from deep_translator import GoogleTranslator
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler, DDIMScheduler
    from controlnet_aux import OpenposeDetector
    from tqdm import tqdm
    from transformers import CLIPVisionModelWithProjection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    print(f"🖥️ Dispositivo selezionato: {device}")

    model_path = f"modelli/{Modelli.get()}"
    model = f"{model_path}.safetensors" if os.path.exists(f"{model_path}.safetensors") else model_path
    if not os.path.exists(model):
        print(f"⚠️ Modello non trovato: {model}")
        return

    is_lcm = "lcm" in model.lower()
    is_pony = "pony" in model.lower()

    prompt_text_raw = prompt.get('1.0', 'end').strip()
    if not prompt_text_raw or prompt_text_raw == 'inserisci prompt positivo':
        print("⚠️ Prompt non valido.")
        return

    neg_prompt = negative.get('1.0', 'end').strip()
    neg_prompt = "" if neg_prompt == 'inserisci prompt negativo' else neg_prompt

    def parse_multiprompt_ranges(text, total_frames):
            prompts = text.split(';')
            parsed = []
            for pr in prompts:
                pr = pr.strip()
                if ':' in pr:
                    try:
                        index, txt = pr.split(':', 1)
                        index = int(index.strip())
                        loras = re.findall(r'<(.*?)>', txt)
                        parsed.append((index, txt.strip(), loras))
                    except ValueError:
                        print(f"⚠️ Errore parsing indice in '{pr}'")
            parsed.sort()
            result = [None] * total_frames
            for i, (start_idx, txt, loras) in enumerate(parsed):
                end_idx = parsed[i+1][0] if i+1 < len(parsed) else total_frames
                for j in range(start_idx, min(end_idx, total_frames)):
                    result[j] = {'image': j, 'prompt': txt, 'loras': loras}
            return result

    try:
        if ';' in prompt_text_raw and ':' in prompt_text_raw:
            prompt_list = parse_multiprompt_ranges(prompt_text_raw, len(framescollection))
        else:
            current_frame = int(scorri_frames.get()) - 1 if scorri_frames.get() else 0
            prompt_list = [{
                'image': current_frame,
                'prompt': prompt_text_raw.rstrip(';'),
                'loras': [Lora.get()] if Lora and Lora.get() not in ('', 'No_Lora') else []
            }]
    except Exception as e:
        print(f"❌ Errore parsing prompt: {e}")
        return

    try:
        translator = GoogleTranslator(source='it', target='en')
        for item in prompt_list:
            try:
                item['translated'] = translator.translate(item['prompt'])
            except:
                item['translated'] = item['prompt']
    except:
        for item in prompt_list:
            item['translated'] = item['prompt']

    try:
        # Definisci il tipo di dati in base al dispositivo
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Carica il modello ControlNet con il tipo di dati specificato
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose", 
            torch_dtype=dtype, 
            use_safetensors=True
        )
        
        processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

        # Carica il modello principale
        try:
            pipe = StableDiffusionControlNetPipeline.from_single_file(
                model, 
                controlnet=controlnet, 
                torch_dtype=dtype
            )
        except:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model, 
                controlnet=controlnet, 
                torch_dtype=dtype
            )

        # Imposta lo scheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
       
        # --- Gestione del dispositivo in modo coerente ---
        if device == "cuda":
            pipe = pipe.to(device)
            pipe.enable_model_cpu_offload()
            pipe.enable_xformers_memory_efficient_attention = True
        else:
            pipe = pipe.to(device)

    except Exception as e:
        print(f"❌ Errore nel caricamento modelli: {e}")
        traceback.print_exc()
        return

    # --- Logica di controllo frame ---
    is_multi_prompt = ';' in prompt_text_raw and ':' in prompt_text_raw
    try:
        scale_value = int(scale.get()) if scale.get() else 1
    except:
        scale_value = 1

    # IP Adapter - Gestione adattamento multiplo
    ip_adapter_enabled = False  # Flag per tracciare se IP Adapter è attivo
    ip_images = []  # Lista vuota per le immagini IP Adapter
    weight_names = []  # Lista per i nomi dei pesi degli adapter
    adapter_scales = []  # Lista per le scale degli adapter

    # Verifica dei percorsi IP Adapter
    face_ip_path = ip_image_path1 if 'ip_image_path1' in globals() else None
    image_ip_path = ip_image_path2 if 'ip_image_path2' in globals() else None
    use_face_adapter = face_ip_path and os.path.exists(face_ip_path)
    use_image_adapter = image_ip_path and os.path.exists(image_ip_path)
    
    # Tentativo di caricare IP Adapter solo se necessario
    if use_face_adapter or use_image_adapter:
        try:
            # Cambia il scheduler a DDIM (richiesto per IP-Adapter)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            
            # Scale per IP adapter - Supporta valori separati per ciascun adapter
            ip_scale_values = ip_adapter_var.get()
            ip_scale_image = float(ip_scale_values)
            ip_scale_face = float(ip_scale_values)
            
            # Configura caricamento di uno o entrambi gli adapter
            model_path = "h94/IP-Adapter"
            
            # Carica encoder condiviso (opzionale, per ottimizzazione)
            try:
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    model_path,
                    subfolder="models/image_encoder",
                    torch_dtype=dtype
                )
                print("✅ Encoder immagini caricato per IP Adapter")
            except Exception as e:
                print(f"⚠️ Non è stato possibile caricare l'encoder specifico: {e}")
                image_encoder = None
            
            # Prepara le liste per immagini e pesi adapter
            if use_image_adapter and use_face_adapter:
                # Entrambi gli adapter
                print("🖼️👤 Utilizzo entrambi gli adapter: immagine completa e volto")
                weight_names = ["ip-adapter_sd15.bin", "ip-adapter-full-face_sd15.bin"]
                
                # Carica le immagini
                full_ip_image = Image.open(image_ip_path).convert("RGB").resize((768, 768), Image.LANCZOS)
                face_ip_image = Image.open(face_ip_path).convert("RGB").resize((768, 768), Image.LANCZOS)
                
                # Assegna le immagini e le scale
                ip_images = [full_ip_image, face_ip_image]
                adapter_scales = [ip_scale_image, ip_scale_face]
                
                # Carica gli adapter
                pipe.load_ip_adapter(
                    model_path,
                    subfolder="models",
                    weight_name=weight_names,
                    image_encoder=image_encoder
                )
                
                print(f"✅ IP Adapter multipli caricati con scale: {adapter_scales}")
                ip_adapter_enabled = True
                
            elif use_image_adapter:
                print("🖼️ Uso immagine completa come riferimento di adapter")
                print(f"📁 Percorso immagine IP Full: {image_ip_path}")
                
                # Carica l'adapter
                pipe.load_ip_adapter(
                    model_path,
                    subfolder="models",
                    weight_name="ip-adapter_sd15.bin",
                    image_encoder=image_encoder,
                    torch_dtype=dtype
                )
                
                # Carica e prepara l'immagine
                full_ip_image = Image.open(image_ip_path).convert("RGB").resize((768, 768), Image.LANCZOS)
                ip_images = [full_ip_image]
                adapter_scales = [ip_scale_image]
                
                ip_adapter_enabled = True
                print(f"✅ IP Adapter Immagine Intera caricato con scala: {ip_scale_image}")
                
            elif use_face_adapter:
                print("👤 Uso immagine del viso come riferimento di adapter")
                print(f"📁 Percorso immagine IP Faccia: {face_ip_path}")
                
                # Carica l'adapter
                pipe.load_ip_adapter(
                    model_path,
                    subfolder="models",
                    weight_name="ip-adapter-full-face_sd15.bin",
                    image_encoder=image_encoder,
                    torch_dtype=dtype
                )
                
                # Carica e prepara l'immagine
                face_ip_image = Image.open(face_ip_path).convert("RGB").resize((768, 768), Image.LANCZOS)
                ip_images = [face_ip_image]
                adapter_scales = [ip_scale_face]
                
                ip_adapter_enabled = True
                print(f"✅ IP Adapter Faccia caricato con scala: {ip_scale_face}")
            
            # Imposta le scale per gli adapter se ci sono più adapter
            if len(adapter_scales) > 0:
                pipe.set_ip_adapter_scale(adapter_scales)
            
            # Assicurati che la pipeline sia sul dispositivo corretto dopo le modifiche
            pipe = pipe.to(device)
            
        except Exception as e:
            print(f"❌ Errore IP Adapter: {e}")
            traceback.print_exc()
            ip_adapter_enabled = False
            ip_images = []

    # Crea directory di output se non esiste
    os.makedirs("./output_image", exist_ok=True)

    for prompt_item in tqdm(prompt_list, desc="Elaborazione prompts"):
        frame_index = prompt_item['image']
        if frame_index < 0 or frame_index >= len(framescollection):
            print(f"⚠️ Frame {frame_index} fuori range, skip.")
            continue

        # --- Condizione di generazione immagine ---
        generate_this_frame = False
        if prompt_text_raw.strip() == ';':
            generate_this_frame = True
        elif is_multi_prompt:
            if frame_index==0 or frame_index % scale_value == 0:
                generate_this_frame = True
        else:
            current_frame = int(scorri_frames.get()) - 1 if scorri_frames.get() else 0
            if frame_index == current_frame:
                generate_this_frame = True

        if not generate_this_frame:
            print(f"⏭️ Skip frame {frame_index}")
            continue

        try:
            current_image = framescollection[frame_index]
            if isinstance(current_image, str) and os.path.exists(current_image):
                current_image = Image.open(current_image).convert("RGB")
            elif not isinstance(current_image, Image.Image):
                print(f"⚠️ Immagine frame {frame_index} non valida, skip.")
                continue
            current_image = current_image.resize((768, 768), Image.BICUBIC)

            # Genera l'immagine di controllo dalle pose
            pose_image = processor(current_image, hand_and_face=True)
            pose_image = pose_image.resize(current_image.size, Image.BILINEAR)

            pose_image.save(f"./output_image/pose_img_{frame_index}.png")

            current_prompt = prompt_item['translated']
            print(f"🖌️ Frame {frame_index}: Prompt: '{current_prompt}'")
            
            # Carica LoRA se necessario
            if prompt_item['loras'] and lorascale:
                for lora_name in prompt_item['loras']:
                    if lora_name != 'No_Lora':
                        lora_path = f"lora/{lora_name}"
                        if os.path.exists(lora_path):
                            try:
                                pipe.load_lora_weights(
                                    lora_path, 
                                    weight_name=lora_name, 
                                    weight=float(lorascale.get())
                                )
                                pipe.to(device)
                            except Exception as e:
                                print(f"⚠️ LoRA non caricata: {e}")

            # Parametri per la generazione text-to-image con ControlNet
            generation_params = {
                "prompt": current_prompt,
                "negative_prompt": neg_prompt,
                "image": pose_image,  # IMPORTANTE: per ControlNet, 'image' è l'immagine di controllo
                "num_inference_steps": int(Steps.get()),
                "guidance_scale": float(Cfg.get()),
                "width": 768,  # Mantiene la dimensione originale
                "height": 768,  # Mantiene la dimensione originale
                "generator": torch.manual_seed(1000) if prompt_text_raw.strip() == ';' else torch.Generator(device=device).manual_seed(torch.seed())
            }
            
            # Aggiungi le immagini IP adapter se disponibili
            if ip_adapter_enabled and ip_images:
                generation_params["ip_adapter_image"] = ip_images

            # Generazione immagine con gestione degli errori
            try:
                result = pipe(**generation_params).images[0]
                result.save(f"./output_image/sd_{frame_index}.png")
                print(f"✅ Immagine salvata: sd_{frame_index}.png")
            except Exception as err:
                print(f"❌ Errore nella generazione dell'immagine: {err}")
                traceback.print_exc()

        except Exception as err:
            print(f"❌ Errore frame {frame_index}: {err}")
            traceback.print_exc()

    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    print("🏁 Generazione completata.")

# Bottone text_to_image
tk.Button(frame_riga1, text="Text_to image", width=25, bg='#00BFFF', fg='black',
          activebackground='#87CEFA', command=text_to_Image).pack(side='left', padx=2)
# Bottone Inpainting (in alto)
# Bottone Inpainting
import numpy as np
import gc, torch, os, re, traceback
from PIL import Image
from transformers import logging
from deep_translator import GoogleTranslator
from diffusers import (
        StableDiffusionControlNetInpaintPipeline,
        ControlNetModel,
        DDIMScheduler,
        UniPCMultistepScheduler
)

from diffusers import FluxTransformer2DModel, FluxImg2ImgPipeline
from transformers import T5EncoderModel, CLIPTextModel
from optimum.quanto import freeze, qfloat8, quantize

def refine_flux(prompt, result, steps, modifica, cfg):
    """
    Funzione per il refinement con Flux
    Args:
        prompt: Prompt di testo
        result: Immagine PIL di input
        steps: Numero di inference steps
        modifica: Strength (0.0-1.0)
        cfg: Guidance scale
    Returns:
        PIL Image refined
    """
    try:
        # LIBERAZIONE COMPLETA VRAM - Rimozione primo pipeline
        print("🗑️ Liberazione VRAM dal primo pipeline...")
        import gc
        import torch
        from diffusers import FluxImg2ImgPipeline
        from diffusers.models import FluxTransformer2DModel
        from transformers import T5EncoderModel, CLIPTextModel
        from optimum.quanto import quantize, freeze, qfloat8
        
        # Verifica se le variabili esistono prima di eliminarle
        globals_to_delete = ['pipe', 'controlnet']
        for var_name in globals_to_delete:
            if var_name in globals():
                del globals()[var_name]
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Attende che tutte le operazioni CUDA finiscano
        print("✅ VRAM liberata completamente")

        bfl_repo = "black-forest-labs/FLUX.1-dev"
        dtype = torch.bfloat16

        # Ottimizzazioni VRAM - Caricamento e quantizzazione transformer
        print("🔄 Caricamento e quantizzazione Transformer...")
        transformer = FluxTransformer2DModel.from_single_file(
            ".//modelli//jibMixFlux_v85Consisteight.safetensors", 
            torch_dtype=dtype
        )
        quantize(transformer, weights=qfloat8)
        freeze(transformer)

        # Ottimizzazioni VRAM - Caricamento e quantizzazione text encoder
        print("🔄 Caricamento e quantizzazione Text Encoder...")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            bfl_repo, 
            subfolder="text_encoder_2", 
            torch_dtype=dtype
        )
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

        # Caricamento CLIP text encoder e quantizzazione
        print("🔄 Caricamento e quantizzazione CLIP Text Encoder...")
        text_encoder = CLIPTextModel.from_pretrained(
            bfl_repo, 
            subfolder="text_encoder", 
            torch_dtype=dtype
        )
        quantize(text_encoder, weights=qfloat8)
        freeze(text_encoder)

        # Creazione pipeline con componenti quantizzati
        print("🔄 Creazione pipeline Flux...")
        pipeflux = FluxImg2ImgPipeline.from_pretrained(
            bfl_repo, 
            transformer=None, 
            text_encoder=None,
            text_encoder_2=None, 
            torch_dtype=dtype
        )

        # Assegnazione componenti quantizzati
        pipeflux.transformer = transformer
        pipeflux.text_encoder = text_encoder
        pipeflux.text_encoder_2 = text_encoder_2

        # Ottimizzazioni VRAM per pipeline Flux
        print("🔄 Applicazione ottimizzazioni VRAM...")
        # Solo model_cpu_offload per compatibilità con quantizzazione
        pipeflux.enable_model_cpu_offload()

        # Ottimizzazioni VAE per risparmio memoria
        if hasattr(pipeflux.vae, 'enable_tiling'):
            pipeflux.vae.enable_tiling()
        if hasattr(pipeflux.vae, 'enable_slicing'):
            pipeflux.vae.enable_slicing()

        # Pulizia memoria CUDA prima della generazione
        torch.cuda.empty_cache()

        print("🎨 Generazione immagine refinement con Flux...")
        wres, hres = result.size

        imageout = pipeflux(
            prompt=prompt,
            image=result,
            width=wres,
            height=hres,
            num_inference_steps=int(steps),
            strength=float(modifica),
            guidance_scale=float(cfg),
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        # Pulizia finale memoria e liberazione pipeline Flux
        print("🗑️ Liberazione finale VRAM...")
        del pipeflux
        del transformer
        del text_encoder
        del text_encoder_2
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        return imageout
        
    except Exception as e:
        print(f"❌ Errore durante il refinement Flux: {e}")
        # Pulizia di emergenza in caso di errore
        torch.cuda.empty_cache()
        return result  # Ritorna l'immagine originale in caso di errore


def Inpainting():
    global Cfg, Steps, framescollection, scorri_frames, prompt, Lora, lorascale, modifica, scale
    global Modelli, negative, ip_adapter_var, maskcollecollection, exadi, ip_image_path1, ip_image_path2
    
    import torch
    import os
    import gc
    import traceback
    import re
    import cv2
    from PIL import Image
    import numpy as np
    from deep_translator import GoogleTranslator
    from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler, DDIMScheduler
    from tqdm import tqdm
    from transformers import CLIPVisionModelWithProjection

    # Verifica dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    print(f"🖥️ Dispositivo selezionato: {device}")

    # Verifica validità modello
    model_name = Modelli.get() if hasattr(Modelli, 'get') else ""
    if not model_name or not ('inp' in model_name.lower() or 'inpainting' in model_name.lower()):
        print(f"⚠️ Modello '{model_name}' non adatto all'Inpainting.")
        return

    model_path = f"modelli/{model_name}"
    model = f"{model_path}.safetensors" if os.path.exists(f"{model_path}.safetensors") else model_path
    if not os.path.exists(model):
        print(f"⚠️ Modello non trovato: {model}")
        return

    # Verifica collezioni di frame e maschere
    if not framescollection or len(framescollection) == 0:
        print("⚠️ Non ci sono frame disponibili. Impossibile procedere.")
        return
    
    if not maskcollecollection or len(maskcollecollection) == 0:
        print("⚠️ Non ci sono maschere disponibili. Impossibile procedere con l'inpainting.")
        return

    # Prompt
    prompt_text_raw = prompt.get('1.0', 'end').strip() if hasattr(prompt, 'get') else ""
    if not prompt_text_raw or prompt_text_raw.lower() == 'inserisci prompt positivo':
        print("⚠️ Prompt non valido.")
        return

    neg_prompt = negative.get('1.0', 'end').strip() if hasattr(negative, 'get') else ""
    neg_prompt = "" if neg_prompt.lower() == 'inserisci prompt negativo' else neg_prompt

    def parse_multiprompt(text, total_frames):
        """
        Parsing stile text-to-image:
        es. '0:prima frase;10:seconda frase;30:terza frase'
        """
        parts = [p.strip() for p in text.split(';') if p.strip()]
        parsed = []
        
        for part in tqdm(parts, desc="Separazione parti del multi prompts"):
            if ':' in part and part.count(':') >= 1:
                try:
                    # Trova il primo ':' per separare indice da prompt
                    colon_index = part.find(':')
                    idx_str = part[:colon_index].strip()
                    txt = part[colon_index + 1:].strip()
                    
                    # Verifica che l'indice sia un numero valido
                    idx = int(idx_str)
                    if 0 <= idx < total_frames and txt:
                        parsed.append((idx, txt))
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Errore parsing parte '{part}': {e}")
                    continue

        # Se non ci sono prompt parsati correttamente, usa il testo originale
        if not parsed:
            lora_name = Lora.get() if hasattr(Lora, 'get') and Lora.get() not in ('', 'No_Lora') else None
            return [{
                'image': i, 
                'prompt': text.strip(), 
                'loras': [lora_name] if lora_name else []
            } for i in range(total_frames)]

        # Ordina per indice
        parsed.sort(key=lambda x: x[0])
        
        # Genera lista completa per tutti i frame
        full_list = []
        for i in tqdm(range(total_frames), desc="Raccoglimento di tutti i prompt per ogni Frame"):
            # Trova il prompt più recente per questo frame
            current_prompt = parsed[0][1]  # Default al primo prompt
            current_loras = []
            
            for frame_idx, frame_prompt in parsed:
                if i >= frame_idx:
                    current_prompt = frame_prompt
                    # Estrai LoRA dal prompt corrente
                    current_loras = re.findall(r'<([^>]+)>', frame_prompt)
                else:
                    break
            
            # Se non ci sono LoRA nel prompt, usa quella globale
            if not current_loras:
                lora_name = Lora.get() if hasattr(Lora, 'get') and Lora.get() not in ('', 'No_Lora') else None
                current_loras = [lora_name] if lora_name else []
            
            full_list.append({
                'image': i,
                'prompt': current_prompt,
                'loras': current_loras
            })
            
        return full_list

    # Determina modalità operativa
    try:
        is_multiprompt = ';' in prompt_text_raw and ':' in prompt_text_raw
        
        if is_multiprompt:
            prompt_list = parse_multiprompt(prompt_text_raw, len(framescollection))
            print(f"📝 Modalità multi-prompt rilevata con {len(prompt_list)} configurazioni")
        else:
            current_frame = 0
            try:
                current_frame = int(scorri_frames.get()) - 1 if hasattr(scorri_frames, 'get') and scorri_frames.get() else 0
                current_frame = max(0, min(current_frame, len(framescollection) - 1))
            except (ValueError, AttributeError):
                current_frame = 0
                
            lora_name = Lora.get() if hasattr(Lora, 'get') and Lora.get() not in ('', 'No_Lora') else None
            prompt_list = [{
                'image': current_frame,
                'prompt': prompt_text_raw.rstrip(';'),
                'loras': [lora_name] if lora_name else []
            }]
            print(f"📝 Modalità prompt singolo per il frame {current_frame}")
            
    except Exception as e:
        print(f"❌ Errore parsing prompt: {e}")
        traceback.print_exc()
        return

    # Traduzione prompt (con gestione errori robusta)
    try:
        translator = GoogleTranslator(source='it', target='en')
        for item in tqdm(prompt_list, desc="Traduzione prompt"):
            try:
                translated = translator.translate(item['prompt'])
                item['translated'] = translated if translated and translated.strip() else item['prompt']
            except Exception as e:
                print(f"⚠️ Errore traduzione per '{item['prompt'][:50]}...': {e}")
                item['translated'] = item['prompt']
    except Exception as e:
        print(f"⚠️ Errore inizializzazione traduttore: {e}")
        for item in prompt_list:
            item['translated'] = item['prompt']

    # Caricamento modelli AI
    try:
        dtype = torch.float16 if device == "cuda" else torch.float32

        print("🔄 Caricamento ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", 
            torch_dtype=dtype
        )

        print("🔄 Caricamento pipeline base...")
        try:
            pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
                model, controlnet=controlnet, torch_dtype=dtype
            )
        except Exception as e:
            print(f"⚠️ Fallback al caricamento da directory: {e}")
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                model, controlnet=controlnet, torch_dtype=dtype
            )

        # Configurazione pipeline
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        
        if device == "cuda":
            pipe.enable_model_cpu_offload()
            # Disabilita xformers se causa problemi
            if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                pipe.enable_xformers_memory_efficient_attention = False
                
    except Exception as e:
        print(f"❌ Errore nel caricamento modelli: {e}")
        traceback.print_exc()
        return

    # Configurazione IP Adapter (con gestione errori migliorata)
    ip_adapter_enabled = False
    ip_images = []
    adapter_scales = []

    # Verifica percorsi IP Adapter con sicurezza
    face_ip_path = globals().get('ip_image_path1', None)
    image_ip_path = globals().get('ip_image_path2', None)
    use_face_adapter = face_ip_path and os.path.exists(face_ip_path)
    use_image_adapter = image_ip_path and os.path.exists(image_ip_path)
    
    if use_face_adapter or use_image_adapter:
        try:
            print("🔄 Configurazione IP Adapter...")
            # Cambia scheduler per IP-Adapter
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            
            # Scala IP adapter
            try:
                ip_scale_value = float(ip_adapter_var.get()) if hasattr(ip_adapter_var, 'get') else 0.5
            except (ValueError, AttributeError):
                ip_scale_value = 0.5
                
            model_path = "h94/IP-Adapter"
            
            # Carica encoder immagini
            try:
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    model_path,
                    subfolder="models/image_encoder",
                    torch_dtype=dtype
                )
                print("✅ Encoder immagini caricato per IP Adapter")
            except Exception as e:
                print(f"⚠️ Encoder specifico non disponibile: {e}")
                image_encoder = None
            
            # Configura adapter basato su immagini disponibili
            if use_image_adapter and use_face_adapter:
                print("🖼️👤 Caricamento entrambi gli adapter")
                weight_names = ["ip-adapter_sd15.bin", "ip-adapter-full-face_sd15.bin"]
                
                full_ip_image = Image.open(image_ip_path).convert("RGB").resize((768, 768), Image.LANCZOS)
                face_ip_image = Image.open(face_ip_path).convert("RGB").resize((768, 768), Image.LANCZOS)
                
                ip_images = [full_ip_image, face_ip_image]
                adapter_scales = [ip_scale_value, ip_scale_value]
                
                pipe.load_ip_adapter(
                    model_path,
                    subfolder="models",
                    weight_name=weight_names,
                    image_encoder=image_encoder
                )
                
            elif use_image_adapter:
                print("🖼️ Caricamento IP Adapter per immagine completa")
                pipe.load_ip_adapter(
                    model_path,
                    subfolder="models",
                    weight_name="ip-adapter_sd15.bin",
                    image_encoder=image_encoder
                )
                
                full_ip_image = Image.open(image_ip_path).convert("RGB").resize((768, 768), Image.LANCZOS)
                ip_images = [full_ip_image]
                adapter_scales = [ip_scale_value]
                
            elif use_face_adapter:
                print("👤 Caricamento IP Adapter per viso")
                pipe.load_ip_adapter(
                    model_path,
                    subfolder="models",
                    weight_name="ip-adapter-full-face_sd15.bin",
                    image_encoder=image_encoder
                )
                
                face_ip_image = Image.open(face_ip_path).convert("RGB").resize((768, 768), Image.LANCZOS)
                ip_images = [face_ip_image]
                adapter_scales = [ip_scale_value]
            
            # Imposta scale adapter
            if adapter_scales:
                pipe.set_ip_adapter_scale(adapter_scales)
                
            pipe = pipe.to(device)
            ip_adapter_enabled = True
            print(f"✅ IP Adapter configurato con scale: {adapter_scales}")
            
        except Exception as e:
            print(f"❌ Errore configurazione IP Adapter: {e}")
            traceback.print_exc()
            ip_adapter_enabled = False
            ip_images = []

    # Funzione per preparare condizione inpainting
    def make_inpaint_condition(image, mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        mask = np.array(mask.convert("L")).astype(np.float32) / 255.0
        image[mask > 0.5] = -1.0
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        return torch.from_numpy(image)

    # Configurazione intervallo frame
    try:
        scale_value = int(scale.get()) if hasattr(scale, 'get') and scale.get() else 1
        scale_value = max(1, scale_value)  # Minimo 1
    except (ValueError, AttributeError):
        scale_value = 1
    
    print(f"🔍 Intervallo frames: ogni {scale_value} frame(s) elaborato(i)")

    # Determina frame da processare
    frames_to_process = []
    if is_multiprompt:
        frames_to_process = [i for i in range(len(framescollection)) if i % scale_value == 0]
        print(f"🎬 Multi-prompt: processando {len(frames_to_process)} frame")
    else:
        current_frame = 0
        try:
            current_frame = int(scorri_frames.get()) - 1 if hasattr(scorri_frames, 'get') else 0
            current_frame = max(0, min(current_frame, len(framescollection) - 1))
        except (ValueError, AttributeError):
            current_frame = 0
        frames_to_process = [current_frame]
        print(f"🎬 Prompt singolo: processando frame {current_frame}")

    # Crea directory output
    os.makedirs("./debug_masks", exist_ok=True)
    os.makedirs("./output_image", exist_ok=True)

    # Crea dizionario prompt per accesso rapido
    prompt_dict = {item['image']: item for item in prompt_list}

    # Elaborazione frame
    for frame_index in tqdm(frames_to_process, desc="Elaborazione frames"):
        # Validazione indici
        if frame_index < 0 or frame_index >= len(framescollection):
            print(f"⚠️ Frame {frame_index} fuori range framescollection, skip.")
            continue
            
        if frame_index >= len(maskcollecollection):
            print(f"⚠️ Frame {frame_index} fuori range maskcollecollection, skip.")
            continue

        # Trova prompt appropriato
        if is_multiprompt:
            # Trova il prompt più recente per questo frame
            applicable_prompts = [(item['image'], item) for item in prompt_list if item['image'] <= frame_index]
            if applicable_prompts:
                prompt_item = max(applicable_prompts, key=lambda x: x[0])[1]
            else:
                prompt_item = prompt_list[0]
        else:
            prompt_item = prompt_list[0]

        try:
            # Caricamento immagine frame
            current_image = framescollection[frame_index]
            if isinstance(current_image, str) and os.path.exists(current_image):
                current_image = Image.open(current_image).convert("RGB")
            elif not isinstance(current_image, Image.Image):
                print(f"⚠️ Immagine frame {frame_index} non valida, skip.")
                continue

            # Caricamento maschera
            raw_mask = maskcollecollection[frame_index]
            if isinstance(raw_mask, np.ndarray):
                raw_mask = Image.fromarray(raw_mask)
            elif isinstance(raw_mask, str) and os.path.exists(raw_mask):
                raw_mask = Image.open(raw_mask)
            elif not isinstance(raw_mask, Image.Image):
                print(f"⚠️ Maschera frame {frame_index} non valida, skip.")
                continue

            mask_image = raw_mask.convert("L")

            # Espansione maschera
            try:
                mask_scale = int(exadi.get()) if hasattr(exadi, 'get') and exadi.get() else 0
                mask_scale = max(0, mask_scale)
            except (ValueError, AttributeError):
                mask_scale = 0

            if mask_scale > 0:
                mask_np = np.array(mask_image)
                _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                
                kernel_size = max(1, mask_scale * 2 + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                expanded_mask = cv2.dilate(binary_mask, kernel, iterations=1)
                
                expanded_mask_image = Image.fromarray(expanded_mask)
                expanded_mask_image.save(f"./debug_masks/expanded_mask_{frame_index}.png")
                mask_image = expanded_mask_image

            # Resize proporzionale
            def resize_proportional(img, target_size=768):
                w, h = img.size
                if w >= h:
                    new_w = target_size
                    new_h = int((target_size * h) / w)
                else:
                    new_h = target_size
                    new_w = int((target_size * w) / h)
                
                # Assicura dimensioni multiple di 8
                new_w = (new_w // 8) * 8
                new_h = (new_h // 8) * 8
                
                return img.resize((new_w, new_h), Image.BICUBIC)

            current_image = resize_proportional(current_image)
            mask_image = resize_proportional(mask_image)
            control_image = make_inpaint_condition(current_image, mask_image)

            current_prompt = prompt_item['translated']
            print(f"🖌️ Frame {frame_index}: '{current_prompt}")

            # Gestione LoRA
            lora_loaded = False
            if prompt_item['loras'] and hasattr(lorascale, 'get'):
                try:
                    lora_scale_value = float(lorascale.get())
                    for lora_name in prompt_item['loras']:
                        if lora_name and lora_name != 'No_Lora':
                            lora_path = f"lora/{lora_name}"
                            if os.path.exists(lora_path):
                                pipe.load_lora_weights(lora_path, adapter_name=lora_name)
                                pipe.set_adapters([lora_name], adapter_weights=[lora_scale_value])
                                lora_loaded = True
                                print(f"✅ LoRA caricata: {lora_name} (scala: {lora_scale_value})")
                                break
                except Exception as e:
                    print(f"⚠️ Errore caricamento LoRA: {e}")

            # Calcolo dimensioni finali
            w, h = current_image.size
            target_size = 1024
            if w >= h:
                new_w = target_size
                new_h = int((target_size * h) / w)
            else:
                new_h = target_size
                new_w = int((target_size * w) / h)
            
            # Arrotonda a multipli di 8
            new_w = (new_w // 8) * 8
            new_h = (new_h // 8) * 8

            # Parametri generazione
            generation_params = {
                "prompt": current_prompt,
                "negative_prompt": neg_prompt,
                "image": current_image,
                "mask_image": mask_image,
                "control_image": control_image,
                "num_inference_steps": int(Steps.get()) if hasattr(Steps, 'get') else 20,
                "guidance_scale": float(Cfg.get()) if hasattr(Cfg, 'get') else 7.5,
                "strength": float(modifica.get()) if hasattr(modifica, 'get') else 1.0,
                "width": new_w,
                "height": new_h,
                "generator": torch.Generator(device=device).manual_seed(42),
            }

            # Aggiungi IP adapter se disponibile
            if ip_adapter_enabled and ip_images:
                generation_params["ip_adapter_image"] = ip_images

            # Generazione
            print(f"🎨 Generazione per frame {frame_index}...")
            result = pipe(**generation_params).images[0]
            output_path = f"./output_image/inpainted_{frame_index}.png"
            result.save(output_path)
            print(f"✅ Immagine salvata: {output_path}")

            # Scarica LoRA dopo l'uso
            if lora_loaded:
                try:
                    pipe.unload_lora_weights()
                except:
                    pass

        except Exception as err:
            print(f"❌ Errore elaborazione frame {frame_index}: {err}")
            traceback.print_exc()
            continue

    # Pulizia finale
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        print("🏁 Elaborazione completata e memoria liberata.")
    except Exception as e:
        print(f"⚠️ Errore pulizia finale: {e}")

    print("🏁 Generazione completata con successo!")


# Pulsante per l'interfaccia (assumendo che tk e frame_riga1 siano definiti altrove)
tk.Button(frame_riga1, text="Inpainting", width=15, bg='#a85ac8', fg='black',
           activebackground='#87CEFA', command=Inpainting).pack(side='left', padx=2)





import os
import numpy as np
from PIL import Image
import torch
from diffusers import FluxTransformer2DModel, FluxFillPipeline
from transformers import T5EncoderModel, CLIPTextModel
from optimum.quanto import freeze, qfloat8, quantize
from tryon_inference import run_inference  
import platform
import shutil
from deep_translator import GoogleTranslator

def F_Flux_Fill():
    """
    Funzione F_Flux_Fill modificata con variabili globali e gestione multi-prompt
    """
    global Cfg, Steps, framescollection, scorri_frames, prompt, Lora, lorascale, modifica, scale
    global Modelli, negative, ip_adapter_var, maskcollecollection, exadi, ip_image_path1, ip_image_path2
    
    import os
    import numpy as np
    from PIL import Image
    import torch
    from diffusers import FluxTransformer2DModel, FluxFillPipeline
    from transformers import T5EncoderModel, CLIPTextModel
    from optimum.quanto import freeze, qfloat8, quantize
    from tryon_inference import run_inference  
    import platform
    import shutil
    import cv2
    import gc
    import traceback
    import re
    from tqdm import tqdm
    

    # Verifica dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    print(f"🖥️ Dispositivo selezionato: {device}")

    # Verifica collezioni di frame e maschere
    if not framescollection or len(framescollection) == 0:
        print("⚠️ Non ci sono frame disponibili. Impossibile procedere.")
        return
    
    if not maskcollecollection or len(maskcollecollection) == 0:
        print("⚠️ Non ci sono maschere disponibili. Impossibile procedere.")
        return

    # Prompt
    prompt_text_raw = prompt.get('1.0', 'end').strip() if hasattr(prompt, 'get') else ""
    if not prompt_text_raw or prompt_text_raw.lower() == 'inserisci prompt positivo':
        print("⚠️ Prompt non valido.")
        return

    neg_prompt = negative.get('1.0', 'end').strip() if hasattr(negative, 'get') else ""
    neg_prompt = "" if neg_prompt.lower() == 'inserisci prompt negativo' else neg_prompt

    def parse_multiprompt(text, total_frames):
        """
        Parsing stile text-to-image:
        es. '0:prima frase;10:seconda frase;30:terza frase'
        """
        parts = [p.strip() for p in text.split(';') if p.strip()]
        parsed = []
        
        for part in tqdm(parts, desc="Separazione parti del multi prompts"):
            if ':' in part and part.count(':') >= 1:
                try:
                    # Trova il primo ':' per separare indice da prompt
                    colon_index = part.find(':')
                    idx_str = part[:colon_index].strip()
                    txt = part[colon_index + 1:].strip()
                    
                    # Verifica che l'indice sia un numero valido
                    idx = int(idx_str)
                    if 0 <= idx < total_frames and txt:
                        parsed.append((idx, txt))
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Errore parsing parte '{part}': {e}")
                    continue

        # Se non ci sono prompt parsati correttamente, usa il testo originale
        if not parsed:
            lora_name = Lora.get() if hasattr(Lora, 'get') and Lora.get() not in ('', 'No_Lora') else None
            return [{
                'image': i, 
                'prompt': text.strip(), 
                'loras': [lora_name] if lora_name else []
            } for i in range(total_frames)]

        # Ordina per indice
        parsed.sort(key=lambda x: x[0])
        
        # Genera lista completa per tutti i frame
        full_list = []
        for i in tqdm(range(total_frames), desc="Raccoglimento di tutti i prompt per ogni Frame"):
            # Trova il prompt più recente per questo frame
            current_prompt = parsed[0][1]  # Default al primo prompt
            current_loras = []
            
            for frame_idx, frame_prompt in parsed:
                if i >= frame_idx:
                    current_prompt = frame_prompt
                    # Estrai LoRA dal prompt corrente
                    current_loras = re.findall(r'<([^>]+)>', frame_prompt)
                else:
                    break
            
            # Se non ci sono LoRA nel prompt, usa quella globale
            if not current_loras:
                lora_name = Lora.get() if hasattr(Lora, 'get') and Lora.get() not in ('', 'No_Lora') else None
                current_loras = [lora_name] if lora_name else []
            
            full_list.append({
                'image': i,
                'prompt': current_prompt,
                'loras': current_loras
            })
            
        return full_list

    def initialize_model():
        """Initialize the FLUX model"""
        # Pulizia cache solo se su Linux e il path esiste
        if platform.system() == "Linux" and os.path.exists("/data-nvme/zerogpu-offload/"):
            try:
                shutil.rmtree("/data-nvme/zerogpu-offload/")
                os.makedirs("/data-nvme/zerogpu-offload/", exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not clean cache directory: {e}")
        
        dtype = torch.bfloat16
        bfl_repo = "black-forest-labs/FLUX.1-dev"
        print(f"Using device: {device}")

        print('Loading diffusion model ...')
        try:
            transformer = FluxTransformer2DModel.from_pretrained(
                "xiaozaa/catvton-flux-alpha", 
                torch_dtype=dtype
            )
            quantize(transformer, weights=qfloat8)
            freeze(transformer)

            text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
            quantize(text_encoder_2, weights=qfloat8)
            freeze(text_encoder_2)

            pipe = FluxFillPipeline.from_pretrained(
                bfl_repo,
                transformer=None, 
                text_encoder_2=None,
                torch_dtype=dtype
            ) 

            pipe.transformer = transformer
            pipe.text_encoder_2 = text_encoder_2
            pipe.enable_model_cpu_offload()
            print('Loading Finished!')
            
            return pipe
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    # Determina modalità operativa
    try:
        is_multiprompt = ';' in prompt_text_raw and ':' in prompt_text_raw
        
        if is_multiprompt:
            prompt_list = parse_multiprompt(prompt_text_raw, len(framescollection))
            print(f"📝 Modalità multi-prompt rilevata con {len(prompt_list)} configurazioni")
        else:
            current_frame = 0
            try:
                current_frame = int(scorri_frames.get()) - 1 if hasattr(scorri_frames, 'get') and scorri_frames.get() else 0
                current_frame = max(0, min(current_frame, len(framescollection) - 1))
            except (ValueError, AttributeError):
                current_frame = 0
                
            lora_name = Lora.get() if hasattr(Lora, 'get') and Lora.get() not in ('', 'No_Lora') else None
            prompt_list = [{
                'image': current_frame,
                'prompt': prompt_text_raw.rstrip(';'),
                'loras': [lora_name] if lora_name else []
            }]
            print(f"📝 Modalità prompt singolo per il frame {current_frame}")
            
    except Exception as e:
        print(f"❌ Errore parsing prompt: {e}")
        traceback.print_exc()
        return

    # Traduzione prompt (con gestione errori robusta)
    try:
        translator = GoogleTranslator(source='it', target='en')
        for item in tqdm(prompt_list, desc="Traduzione prompt"):
            try:
                translated = translator.translate(item['prompt'])
                item['translated'] = translated if translated and translated.strip() else item['prompt']
            except Exception as e:
                print(f"⚠️ Errore traduzione per '{item['prompt'][:50]}...': {e}")
                item['translated'] = item['prompt']
    except Exception as e:
        print(f"⚠️ Errore inizializzazione traduttore: {e}")
        for item in prompt_list:
            item['translated'] = item['prompt']

    # Inizializza il modello
    pipe = initialize_model()

    # Configurazione intervallo frame
    try:
        scale_value = int(scale.get()) if hasattr(scale, 'get') and scale.get() else 1
        scale_value = max(1, scale_value)  # Minimo 1
    except (ValueError, AttributeError):
        scale_value = 1
    
    print(f"🔍 Intervallo frames: ogni {scale_value} frame(s) elaborato(i)")

    # Determina frame da processare
    frames_to_process = []
    if is_multiprompt:
        frames_to_process = [i for i in range(len(framescollection)) if i % scale_value == 0]
        print(f"🎬 Multi-prompt: processando {len(frames_to_process)} frame")
    else:
        current_frame = 0
        try:
            current_frame = int(scorri_frames.get()) - 1 if hasattr(scorri_frames, 'get') else 0
            current_frame = max(0, min(current_frame, len(framescollection) - 1))
        except (ValueError, AttributeError):
            current_frame = 0
        frames_to_process = [current_frame]
        print(f"🎬 Prompt singolo: processando frame {current_frame}")

    # Crea directory output
    os.makedirs("./debug_masks", exist_ok=True)
    os.makedirs("./output_flux_fill", exist_ok=True)
    os.makedirs("./temp_flux", exist_ok=True)

    # Determina reference_face_path (priorità: ip_image_path1, poi ip_image_path2)
    reference_face_path = None
    if ip_image_path1 and os.path.exists(ip_image_path1):
        reference_face_path = ip_image_path1
        print(f"👤 Usando ip_image_path1 come reference face: {ip_image_path1}")
    elif ip_image_path2 and os.path.exists(ip_image_path2):
        reference_face_path = ip_image_path2
        print(f"👤 Usando ip_image_path2 come reference face: {ip_image_path2}")
    else:
        print("⚠️ Nessun reference face path valido trovato")

    # Elaborazione frame
    for frame_index in tqdm(frames_to_process, desc="Elaborazione frames"):
        # Validazione indici
        if frame_index < 0 or frame_index >= len(framescollection):
            print(f"⚠️ Frame {frame_index} fuori range framescollection, skip.")
            continue
            
        if frame_index >= len(maskcollecollection):
            print(f"⚠️ Frame {frame_index} fuori range maskcollecollection, skip.")
            continue

        # Trova prompt appropriato
        if is_multiprompt:
            # Trova il prompt più recente per questo frame
            applicable_prompts = [(item['image'], item) for item in prompt_list if item['image'] <= frame_index]
            if applicable_prompts:
                prompt_item = max(applicable_prompts, key=lambda x: x[0])[1]
            else:
                prompt_item = prompt_list[0]
        else:
            prompt_item = prompt_list[0]

        try:
            # Caricamento immagine frame (da dati in memoria)
            current_image_data = framescollection[frame_index]
            if isinstance(current_image_data, np.ndarray):
                current_image = Image.fromarray(current_image_data)
            elif isinstance(current_image_data, Image.Image):
                current_image = current_image_data
            elif isinstance(current_image_data, str) and os.path.exists(current_image_data):
                current_image = Image.open(current_image_data).convert("RGB")
            else:
                print(f"⚠️ Immagine frame {frame_index} non valida, skip.")
                continue

            # Salva immagine temporanea
            w, h = current_image.size
            temp_image_path = f"./temp_flux/temp_frame_{frame_index}.jpg"
            current_image.save(temp_image_path)

            # Caricamento maschera (da dati in memoria)
            raw_mask_data = maskcollecollection[frame_index]
            if isinstance(raw_mask_data, np.ndarray):
                raw_mask = Image.fromarray(raw_mask_data)
            elif isinstance(raw_mask_data, Image.Image):
                raw_mask = raw_mask_data
            elif isinstance(raw_mask_data, str) and os.path.exists(raw_mask_data):
                raw_mask = Image.open(raw_mask_data)
            else:
                print(f"⚠️ Maschera frame {frame_index} non valida, skip.")
                continue

            mask_image = raw_mask.convert("L")

            # Espansione maschera
            try:
                mask_scale = int(exadi.get()) if hasattr(exadi, 'get') and exadi.get() else 0
                mask_scale = max(0, mask_scale)
            except (ValueError, AttributeError):
                mask_scale = 0

            if mask_scale > 0:
                mask_np = np.array(mask_image)
                _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                
                kernel_size = max(1, mask_scale * 2 + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                expanded_mask = cv2.dilate(binary_mask, kernel, iterations=1)
                
                expanded_mask_image = Image.fromarray(expanded_mask)
                expanded_mask_image.save(f"./debug_masks/expanded_mask_{frame_index}.png")
                mask_image = expanded_mask_image

            # Salva maschera temporanea
            temp_mask_path = f"./temp_flux/temp_mask_{frame_index}.jpg"
            mask_image.save(temp_mask_path)

            # Determina garment path
            garment_path = reference_face_path if reference_face_path else temp_image_path

            current_prompt = prompt_item['translated']
            print(f"🖌️ Frame {frame_index}: '{current_prompt[:100]}...'")

            # Parametri di generazione dalle variabili globali
            try:
                num_steps = int(Steps.get()) if hasattr(Steps, 'get') else 20
                guidance_scale = float(Cfg.get()) if hasattr(Cfg, 'get') else 30.0
                strength = float(modifica.get()) if hasattr(modifica, 'get') else 0.50
            except (ValueError, AttributeError):
                num_steps = 20
                guidance_scale = 30.0
                strength = 0.50

            # Esegui l'inference
            print(f"🎨 Elaborazione virtual try-on per frame {frame_index}...")
            print(f"""parametri attuali: 
                  -- Pipe: {pipe}
                  -- prompt: {current_prompt}
                  -- imagine frame: {temp_image_path}
                  -- image mask: {temp_mask_path}
                  -- image riferimento: {garment_path}
                  -- numero steps: {num_steps}
                  -- guidance_scale: int 30 define
                  -- modifica strength: {strength}
                  -- risoluzione elaborazione: (768, 1024) define
                  -- risoluzione image input e output: W:{w},H:{h}""")
            _, tryon_result = run_inference(
                pipe=pipe,
                prompt=current_prompt,
                image_path=temp_image_path,
                mask_path=temp_mask_path,
                garment_path=garment_path,
                num_steps=num_steps,
                guidance_scale=int(30),
                strength=strength,
                seed=-1,  # Random seed
                size=(768, 1024)  # Dimensioni standard per FLUX
            )
            
            # Salva il risultato
            output_path = f"./output_image/flux_inpainting{frame_index}.jpg"
            output_pathtemp = f"./output_image/flux_temp{frame_index}.jpg"
            
            if isinstance(tryon_result, Image.Image):
                tryon_result = tryon_result.resize((w, h), Image.BICUBIC)
                tryon_result.save(output_pathtemp)
                tryon_result_refine= refine_flux(prompt= current_prompt, result= tryon_result, steps=num_steps, modifica=strength, cfg=int(30))
                tryon_result_refine.save(output_path)
                
                print(f"✅ Risultato salvato in: {output_path}")
            else:
                # Se il risultato è un array numpy, convertilo in immagine
                if isinstance(tryon_result, np.ndarray):
                    tryon_result = Image.fromarray(tryon_result.astype(np.uint8))
                    tryon_result = tryon_result.resize((w, h), Image.BICUBIC)
                    tryon_result.save(output_pathtemp)
                    tryon_result_refine= refine_flux(prompt= current_prompt, result= tryon_result, steps=num_steps, modifica=strength, cfg=int(30))
                    tryon_result_refine.save(output_path)
                    print(f"✅ Risultato salvato in: {output_path}")
                else:
                    print(f"⚠️ Formato risultato non riconosciuto: {type(tryon_result)}")

            # Pulizia file temporanei del frame corrente
            for temp_file in [temp_image_path, temp_mask_path]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

        except Exception as err:
            print(f"❌ Errore elaborazione frame {frame_index}: {err}")
            traceback.print_exc()
            continue

    # Pulizia finale
    try:
        # Rimuovi directory temporanea
        if os.path.exists("./temp_flux"):
            shutil.rmtree("./temp_flux")
        
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        print("🏁 Elaborazione completata e memoria liberata.")
    except Exception as e:
        print(f"⚠️ Errore pulizia finale: {e}")

    print("🏁 Generazione Flux Fill completata con successo!")


def run():
    """Funzione principale per eseguire lo script"""
    print("=== FLUX FILL Virtual Try-On (Memory Optimized) ===")
    print(f"Sistema operativo: {platform.system()}")
    print(f"CUDA disponibile: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # Pulisci cache GPU se necessario
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
    
    # Esegui il Flux Fill
    try:
        F_Flux_Fill()
        print("Flux Fill completato con successo!")
        
    except Exception as e:
        print(f"Errore nell'esecuzione: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Pulisci memoria alla fine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Final GPU cache cleanup completed")


    



Button_Refine = tk.Button(frame_riga1, text="Flux_Fill", width=20, bg='#00c746', fg='black',
                          activebackground='#87CEFA', command=F_Flux_Fill)
Button_Refine.pack(side='left', padx=2)

controlNetSelect = ttk.Combobox(frame_riga1, values=['open pose', 'face_tongue'])
controlNetSelect.pack(side='left', padx=2)
controlNetSelect.set('open pose')

tk.Label(frame_riga1,text= 'Numero Frames Flux_IP').pack(side='left')


# ----- RIGA 2: sotto il bottone Inpainting -----
frame_riga2 = tk.Frame(frame_bottoni_superiori)
frame_riga2.pack()

def Load_mask():
    global maskcollecollection  # Dichiara la variabile globale

    path = filedialog.askopenfilename()

    if path.lower().endswith(('.png', '.jpg', '.jpeg')):
        folder = os.path.dirname(path)
        maskcollecollection = []
        for f in os.listdir(folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(folder, f)
                try:
                    img = Image.open(full_path)
                    maskcollecollection.append(img)
                    print(f"✓ Image frame: {full_path}")
                except Exception as e:
                    print(f"⚠️ Errore nel caricamento dell'immagine: {e}")

    elif path.lower().endswith(('.mp4', '.mkv')):
        try:
            print(f"🎞️ Caricamento video: {path}")
            clip = VideoFileClip(path)
            maskcollecollection = [frame for frame in clip.iter_frames()]
            clip.close()
            print(f"✓ {len(maskcollecollection)} frame estratti dal video.")
        except Exception as e:
            print(f"❌ Errore nel caricamento del video: {e}")
    else:
        print("⚠️ Formato file non supportato.")

tk.Button(frame_riga2, text="Load Mask", width=20, bg='#a1a1a1', fg='black',
          activebackground='#87CEFA', command=Load_mask).pack(side='left', padx=2,pady=4)

from transformers import pipeline
from PIL import Image
import numpy as np
import os
from moviepy import ImageSequenceClip, VideoFileClip
from tqdm import tqdm

def rileva_vestiti():
    global filepath, framescollection, scorri_frames

    if filepath is not None and framescollection:
        try:
            segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes", device=0)  # Use device=0 for CUDA

            video_clip = VideoFileClip(filepath)
            video_size = video_clip.size
            fps = video_clip.fps

            primary_labels = ["Upper-clothes", "Skirt", "Pants", "Dress"]
            fallback_label = "Belt"
            optional_label = "Scarf"

            output_dir = os.path.join(os.getcwd(), "masks", os.path.basename(filepath).split('.')[0])
            os.makedirs(output_dir, exist_ok=True)

            # Process frames and save masks
            for k in tqdm(range(len(framescollection)), desc="Creazione maschere rilevamento vestititi"):
                # Check if framescollection[k] is a string (path) and load the image
                if isinstance(framescollection[k], str):
                    img = Image.open(framescollection[k]).convert("RGB")
                else:
                    # Assume it's already a PIL Image object
                    img = framescollection[k].convert("RGB")
                
                segments = segmenter(img)

                found_labels = set()
                mask_list = []

                for s in segments:
                    label = s['label']
                    if label in primary_labels:
                        found_labels.add(label)
                        mask_list.append(s['mask'])

                if "Skirt" not in found_labels and "Pants" not in found_labels:
                    for s in segments:
                        if s['label'] == fallback_label:
                            mask_list.append(s['mask'])

                for s in segments:
                    if s['label'] == optional_label:
                        mask_list.append(s['mask'])

                if not mask_list:
                    print(f"[Frame {k}] Nessun indumento rilevato.")
                    continue

                final_mask = np.zeros_like(np.array(mask_list[0]), dtype=np.uint8)
                for mask in mask_list:
                    final_mask = np.logical_or(final_mask, np.array(mask)).astype(np.uint8) * 255

                final_mask_img = Image.fromarray(final_mask).resize(video_size, Image.NEAREST)

                mask_filename = f"{os.path.basename(filepath).split('.')[0]}_frame_{k}.png"
                mask_path = os.path.join(output_dir, mask_filename)
                final_mask_img.save(mask_path)
            
            # Create the output video path
            video_name = os.path.basename(filepath).split('.')[0]
            output_video_path = os.path.join(output_dir, f"{video_name}_masks.mp4")
            
            # Load masks for video creation
            masks = []
            for k in tqdm(range(len(framescollection)), desc="Caricamento maschere per video"):
                mask_path = os.path.join(output_dir, f"{video_name}_frame_{k}.png")
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('RGB')
                    masks.append(np.array(mask))
            
            # Create video from masks
            if masks:
                print("Creazione video dalle maschere...")
                video = ImageSequenceClip(masks, fps=fps)
                video.write_videofile(output_video_path, codec="libx264")
                print(f"🎥 Video maschere creato: {output_video_path}")
                
                # Clean up temporary files
                for filename in tqdm(os.listdir(output_dir), desc="Eliminazione file temporanei"):
                    if filename.endswith(('.jpg', '.png', '.jpeg')) and filename != f"{video_name}_masks.mp4":
                        file_path = os.path.join(output_dir, filename)
                        os.remove(file_path)
                print("🧹 Maschere temporanee eliminate.")
            else:
                print("⚠️ Nessuna maschera creata, impossibile generare il video.")

        except Exception as e:
            print(f"❌ Errore nel rilevamento: {str(e)}")
    else:
        print("⚠️ File non caricato o frames vuoti.")



# Create button in the UI
tk.Button(frame_riga2, text="Rileva Vestiti", width=20, bg='#07ad8c', fg='black',
          activebackground='#87CEFA', command=rileva_vestiti).pack(side='left', padx=2, pady=4)

def rileva_capelli():
    global framescollection, filepath
    print("Rileva capelli")
    
    # Assicurati che le directory esistano
    os.makedirs("./pytorch-hair-segmentation/photo", exist_ok=True)
    os.makedirs("./pytorch-hair-segmentation/masks", exist_ok=True)
    
    # Elimina file precedenti
    for img in os.listdir("./pytorch-hair-segmentation/photo"):
        os.remove(os.path.join("./pytorch-hair-segmentation/photo", img))
    
    if os.path.exists("./pytorch-hair-segmentation/masks"):
        for img in os.listdir("./pytorch-hair-segmentation/masks"):
            os.remove(os.path.join("./pytorch-hair-segmentation/masks", img))
    
    # Process frames and save masks
    for k in tqdm(range(len(framescollection)), desc="Creazione maschere rilevamento Capelli"):
        img = None
        # Check if framescollection[k] is a string (path) and load the image
        if isinstance(framescollection[k], str):
            img = Image.open(framescollection[k]).convert("RGB")
        else:
            # Assume it's already a PIL Image object
            img = framescollection[k].convert("RGB")
        
        if img is not None:
            img.save(f"./pytorch-hair-segmentation/photo/frame_{k:04d}.png")
    
    # Cambia directory ed esegui lo script di segmentazione
    current_dir = os.getcwd()
    os.chdir("pytorch-hair-segmentation")
    os.system('python demo.py --img_dir "photo" --use_gpu True')
    os.chdir(current_dir)  # Torna alla directory originale
    
    # Crea il video dalle maschere
    masks_dir = "./pytorch-hair-segmentation/masks"
    if os.path.exists(masks_dir):
        mask_files = sorted([os.path.join(masks_dir, img) for img in os.listdir(masks_dir) if img.endswith('.png')])
        
        if mask_files:  # Verifica che ci siano file nella lista
            # Crea la directory per il video di output
            output_dir = f"masks/{os.path.splitext(os.path.basename(filepath))[0]}"
            os.makedirs(output_dir, exist_ok=True)
            
           
            
            # Carica le immagini come frames
            frames = []
            for mask_path in mask_files:
                img = Image.open(mask_path).convert("RGB")  # Converte esplicitamente in RGB
                img_array = np.array(img)
                
                # Verifica che l'immagine abbia la forma corretta
                if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                    frames.append(img_array)
                else:
                    # Se l'immagine è in scala di grigi, convertila in RGB
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array, img_array, img_array], axis=2)
                    frames.append(img_array)
            
            # Verifica che ci siano frame validi
            if not frames:
                print("Nessun frame valido per creare il video")
                return
                
            # Crea e salva il video
            output_path = f"{output_dir}/{os.path.splitext(os.path.basename(filepath))[0]}.mp4"
            try:
                video = ImageSequenceClip(frames, fps=8)
                video.write_videofile(output_path, codec='libx264', fps=8)
                print(f"Video salvato in: {output_path}")
            except Exception as e:
                print(f"Errore nella creazione del video: {e}")
            
            print(f"Video salvato in: {output_path}")
        else:
            print("Nessuna maschera trovata nella directory")
    else:
        print(f"La directory {masks_dir} non esiste")



tk.Button(frame_riga2,text= "Rileva Capeli",width=20, bg='#43acff', fg='black',
          activebackground='#87CEFA', command=rileva_capelli).pack(side='left', padx=2, pady=4)


import torch
from diffusers import I2VGenXLPipeline
from PIL import Image
import tkinter as tk
from deep_translator import GoogleTranslator  # Assumendo che usi deep_translator

def F_SVD_XT():
    global Cfg, Steps, framescollection, scorri_frames, prompt, negative

    # Caricamento immagine
    try:
        index = int(scorri_frames.get())
        image = Image.open(framescollection[index])
        print(f"✅ Immagine caricata: {image.size}")
    except Exception as e:
        print(f"❌ Errore caricamento immagine: {e}")
        return

    # Ridimensionamento immagine e arrotondamento a multipli di 8
    risoluzione = 960
    wi, hi = image.size
    if wi >= hi:
        new_height = (risoluzione * hi) // wi
        new_height -= new_height % 8  # Arrotonda a multiplo di 8
        width, height = risoluzione, new_height
    else:
        new_width = (risoluzione * wi) // hi
        new_width -= new_width % 8
        width, height = new_width, risoluzione

    image = image.resize((width, height))

    # Traduzione prompt
    # Corretto 'sorce' -> 'source', 'traslate' -> 'translate'
    prompt_text = GoogleTranslator(source='italian', target='english').translate(prompt.get('1.0', tk.END).strip())

    negative_prompt = negative.get().strip()

    # Caricamento pipeline
    pipe = I2VGenXLPipeline.from_pretrained(
        "ali-vilab/i2vgen-xl", torch_dtype=torch.bfloat16, variant="fp16"
    )
    pipe.enable_model_cpu_offload()

    print("🎬 Generazione video in corso...")
    generator = torch.manual_seed(0)
    fps = 8

    # Generazione frames
    result = pipe(
        prompt=prompt_text,
        image=image,
        width=width,
        height=height,
        num_inference_steps=int(Steps.get()),
        negative_prompt=negative_prompt,
        guidance_scale=float(Cfg.get()),
        generator=generator,
    )

    frames = result.frames[0]

    # Esportazione video (devi aver definito questa funzione)
    export_to_video(frames, "i2v.mp4", fps=fps)
    print("✅ Video generato: i2v.mp4")

        
        
        




tk.Button(frame_riga2,text= "SVD_XT",width=20, bg='#986caf', fg='black',
          activebackground='#87CEFA', command=F_SVD_XT).pack(side='left', padx=2, pady=4)
        
# General Libraries for the program's functionality
import os
import gc
import torch
from PIL import Image
from tqdm import tqdm

# Import both text encoders at the top
from transformers import T5EncoderModel, CLIPTextModel
from optimum.quanto import freeze, qfloat8, quantize
from deep_translator import GoogleTranslator
import tkinter as tk
import glob
import random

# Diffusers libraries for ControlNet (keep these as they are)
from diffusers import FluxPipeline, FluxControlNetPipeline, FluxControlNetModel, FluxMultiControlNetModel
# Import FluxTransformer2DModel at the top, it's always needed
from diffusers import FluxTransformer2DModel

def F_flux_adapter():
    import os
    import torch
    from PIL import Image
    from diffusers import FluxPipeline, FluxTransformer2DModel
    from transformers import T5EncoderModel
    from optimum.quanto import qfloat8, quantize, freeze
    from deep_translator import GoogleTranslator

    global prompt, ip_image_path1, ip_image_path2, Steps, Cfg, risoluzione, Modelli, Lora, lorascale, modifica, numero_di_riperizioni, ip_adapter_scale

    model_or = "black-forest-labs/FLUX.1-dev"
    model_id_nsfw = "trongg/FLUX.1-dev_nsfw_FLUXTASTIC-v3.0"
    model_id= model_or 
    dtype = torch.bfloat16
    IP_ok = False
    image_path = "immagine_ref.jpg"

    if "flux.1-dev_nsfw" in Modelli.get():
        model_id= model_id_nsfw
    else:
        model_id=model_or
    

    def ridimensiona_a_512(img):
        w, h = img.size
        if w >= h:
            new_h = (512 * h) // w
            img = img.resize((512, new_h), Image.BICUBIC)
        else:
            new_w = (512 * w) // h
            img = img.resize((new_w, 512), Image.BICUBIC)
        return img

    # === Composizione immagine IP se presenti due path ===
    # === Gestione immagine di riferimento IP-Adapter ===
    if ip_image_path1 and os.path.exists(ip_image_path1) and ip_image_path2 and os.path.exists(ip_image_path2):
        image1 = ridimensiona_a_512(Image.open(ip_image_path1))
        image2 = ridimensiona_a_512(Image.open(ip_image_path2))
        new_image = Image.new("RGB", ((image1.width + image2.width), max(image1.height, image2.height)), "white")
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1.width, 0))
        new_image.save("immagine_ref.jpg")
        IP_ok = True

    elif ip_image_path1 and os.path.exists(ip_image_path1):
        new_image = ridimensiona_a_512(Image.open(ip_image_path1))
        new_image.save("immagine_ref.jpg")
        IP_ok = True

    elif ip_image_path2 and os.path.exists(ip_image_path2):
        new_image = ridimensiona_a_512(Image.open(ip_image_path2))
        new_image.save("immagine_ref.jpg")
        IP_ok = True

    else:
        IP_ok = False

    # === Caricamento pipeline principale ===
    print(f"🔄 Caricamento della pipeline... Model: {model_id}")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)

    # === Quantizzazione transformer ===
    print("🔄 Caricamento e quantizzazione del transformer...")
    transformer = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=dtype)
    quantize(transformer, weights=qfloat8)
    freeze(transformer)

    # === Quantizzazione text_encoder_2 ===
    print("🔄 Caricamento e quantizzazione di text_encoder_2...")
    text_encoder_2 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype)
    quantize(text_encoder_2, weights=qfloat8)
    freeze(text_encoder_2)

    # === Caricamento LoRA se selezionata ===
    if "flux" in Lora.get().lower():
        print("📎 Caricamento della LoRA...")
        lorapath = f"./Lora/{Lora.get()}"
        if not lorapath.endswith(".safetensors"):
            lorapath += ".safetensors"
        adapter_name = os.path.splitext(os.path.basename(lorapath))[0]
        pipe.load_lora_weights(lorapath, weight_name=os.path.basename(lorapath), adapter_name=adapter_name)
        pipe.set_adapters(adapter_name, adapter_weights=float(lorascale.get()))
        print("✅ Adapters attivi:", pipe.get_active_adapters())

    print("🔁 Riassegno i moduli quantificati post-LoRA")
    pipe.transformer = transformer
    pipe.text_encoder_2 = text_encoder_2

    if IP_ok==True:
        # === Caricamento IP-Adapter ===
        print("📎 Caricamento dell'IP-Adapter...")
        pipe.load_ip_adapter(
            "XLabs-AI/flux-ip-adapter",
            weight_name="ip_adapter.safetensors",
            image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
        )
        pipe.set_ip_adapter_scale(float(ip_adapter_scale.get()))

    # === Offload automatico ===
    pipe.enable_model_cpu_offload()

    # === Caricamento immagine riferimento ===
    image = Image.open(image_path)
    print(f"🖼️ Riferimento IP-Adapter: {image.size}")

    # === Traduzione prompt ===
    prompt_text = prompt.get('1.0', 'end').strip()
    prompt_engl = GoogleTranslator(source="it", target="en").translate(prompt_text)
    for n in range(1,int(numero_di_riperizioni.get())):
        # === Generazione ===
        print("🎨 Generazione immagine...")
        import random
        seed = random.randrange(1, 100000)
        print(f"🧪 Seed casuale usato: {seed}")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        width, height = map(int, risoluzione.get().split(','))
        steps = int(Steps.get())
        if IP_ok== True:
                images = pipe(
                    prompt=prompt_engl,
                    width=width,
                    height=height,
                    guidance_scale=float(Cfg.get()),
                    num_inference_steps=steps,
                    generator=generator,
                    ip_adapter_image=image,
                    max_sequence_length=512,
                ).images
        else:
                images = pipe(
                    prompt=prompt_engl,
                    width=width,
                    height=height,
                    guidance_scale=float(Cfg.get()),
                    num_inference_steps=steps,
                    generator=generator,
                    max_sequence_length=512,
                ).images

        # === Salvataggio risultato ===
        print("💾 Salvataggio immagine...")
        os.makedirs("./output_image", exist_ok=True)
        images[0] = images[0].convert("RGB")
        if IP_ok==True:
            images[0].save(f"./output_image/sd_flux_IP_{n}.jpg")
        else:
            images[0].save(f"./output_image/sd_flux_{n}.jpg")
        print("✅ Immagine salvata come './output_image/sd_flux_IP.jpg'")


   





            
    


tk.Button(frame_riga2,text= "Flux_IP_adapter",width=20, bg='#e8ff6a', fg='black',
          activebackground='#87CEFA', command=F_flux_adapter).pack(side='left', padx=2, pady=4)
        
numero_di_riperizioni = ttk.Combobox(frame_riga2, values=list(range(1, 21)))
numero_di_riperizioni.pack(side='left', padx=2, pady=4)
numero_di_riperizioni.set(1)

# Frame per intervallo e bottone interpolazione
frame_intervallo_interpola = tk.Frame(frame_strumenti)
frame_intervallo_interpola.pack(pady=5, fill='x')

# Sezione intervallo a sinistra
frame_intervallo = tk.Frame(frame_intervallo_interpola)
frame_intervallo.pack(side='left', padx=2)

intervallo_var = tk.IntVar(value=1)

def aggiorna_intervallo(val=None):
    intervallo_label.config(text=f"Intervallo: {intervallo_var.get()},10")

intervallo_label = tk.Label(frame_intervallo, text="Intervallo: 1,10", font=("Arial", 10))
intervallo_label.pack(side='top', anchor='w')

scale = tk.Scale(frame_intervallo, width=15, from_=1, to=10, resolution=1,
               variable=intervallo_var, orient='horizontal',
               command=aggiorna_intervallo,
               bg='#f0f0ff', fg='black', troughcolor='#9370DB',
               highlightthickness=0, length=150)
scale.pack(side='bottom')

def interpolazione():
    print("interpolazione")
    path_frames = "./frame-interpolation-pytorch/photos"
    
    # Pulisce la cartella di destinazione
    for f in os.listdir(path_frames):
        file_path = os.path.join(path_frames, f)
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Copia solo i file desiderati
    for f in os.listdir('output_image'):
        if 'sd_' in f or 'inpainted_' in f or '_inpainting' in f:
            shutil.copyfile(os.path.join('output_image', f), os.path.join(path_frames, f))
    
    # Esegue lo script di interpolazione
    os.chdir('frame-interpolation-pytorch')
    os.system("python inter_poly.py")
    os.chdir('..')
    
    # Sposta il video se è stato creato
    if os.path.exists("frame-interpolation-pytorch/videoAnimazione.mp4"):
        shutil.move("frame-interpolation-pytorch/videoAnimazione.mp4", "enhancement/video_inerpolato.mp4")

button_interpola = tk.Button(frame_intervallo_interpola, text="interola frame", width=25,
                            command=interpolazione, relief='raised', bg='green', fg='black')
button_interpola.pack(side='left', padx=6)

frame_apri_file = tk.Frame(frame_strumenti)
frame_apri_file.pack(pady=5, fill='x')
tk.Button(frame_apri_file, text='Apri file video/Frames', command=aprifile,
         width=52, bg='gray', fg='white', activebackground='#696969').pack(pady=5)

# Frame per IP adapter e canvas
frame_ip_canvas = tk.Frame(frame_strumenti)
frame_ip_canvas.pack(pady=5, fill='x')

# Sezione IP adapter a sinistra
frame_ip = tk.Frame(frame_ip_canvas)
frame_ip.pack(side='left', padx=2)

ip_adapter_var = tk.DoubleVar(value=0.5)  # Valore di default 0.5

def aggiorna_ip_adapter(val=None):
    ip_label.config(text=f"Ip Adapter: {ip_adapter_var.get():.1f},1.0")

ip_label = tk.Label(frame_ip, text="Ip Adapter: 0.5,1.0", font=("Arial", 10))
ip_label.pack(side='top', anchor='w')

ip_adapter_scale = tk.Scale(frame_ip, width=15, from_=0.0, to=1.0, resolution=0.1,
                           variable=ip_adapter_var, orient='horizontal',
                           command=aggiorna_ip_adapter,
                           bg='#f0f0ff', fg='black', troughcolor='#9370DB',
                           highlightthickness=0, length=150)
ip_adapter_scale.pack()  # Pack normally without side='bottom'

# Creiamo due frame separati per i canvas
frame_canvas1 = tk.Frame(frame_ip_canvas, bg='blue', width=250, height=150)
frame_canvas1.pack(side='left', padx=2)
frame_canvas1.pack_propagate(False)  # Impedisce al frame di ridimensionarsi in base ai contenuti

frame_canvas2 = tk.Frame(frame_ip_canvas, bg='blue', width=250, height=150)
frame_canvas2.pack(side='left', padx=2)
frame_canvas2.pack_propagate(False)

# Canvas per la prima immagine
canvas = tk.Canvas(frame_canvas1, bg='blue', highlightthickness=0)
canvas.pack(expand=True, fill='both')
canvas_text = canvas.create_text(125, 75, text="IP Adapter:\nImmagine Faccia",
                                font=("Arial", 14), fill='white')

# Canvas per la seconda immagine
canvas2 = tk.Canvas(frame_canvas2, bg='blue', highlightthickness=0)
canvas2.pack(expand=True, fill='both')
canvas_text2 = canvas2.create_text(125, 75, text="IP Adapter:\nImmagine Intera",
                                 font=("Arial", 14), fill='white')
# Variabili globali per le immagini e i percorsi
current_image1 = None
photo_image1 = None
current_image2 = None
photo_image2 = None
ip_image_path1 = None
ip_image_path2 = None




# Crea un nuovo frame per i bottoni sotto il frame_ip_canvas
frame_text_video = tk.Frame(frame_strumenti)  # Usa frame_strumenti come genitore
frame_text_video.pack(pady=(5, 0), fill='x')  # Posizionalo sotto il frame_ip_canvas

from huggingface_hub import snapshot_download

import os
import time

def FramePack_ImagetoVideo():
    print("Running FramePack Image to Video...")
    os.chdir('FramePack')
    os.system("python demo_gradio.py --inbrowser")  # Lancia e apre browser
    # Non serve più webbrowser.open() se --inbrowser funziona correttamente
   

from PIL import Image


def FramePack_ImageToVideo_F1():
    print("FramePack Image to Video F1")
    os.chdir('FramePack')
    os.system("python demo_gradio_f1.py --inbrowser")  # Lancia e apre browser
    # Non serve più webbrowser.open() se --inbrowser funziona correttamente
     

# Utilizza un frame interno per i due bottoni
buttons_frame = tk.Frame(frame_text_video)
buttons_frame.pack()

# Aggiungi i due bottoni affiancati
Text_to_VideoLTX = tk.Button(buttons_frame, text="FramePack ImageToVideo",
                          command=FramePack_ImagetoVideo, bg='#00c4bd', fg='black')
Text_to_VideoLTX.pack(side='left', padx=(0, 5))  # Aggiunge spazio a destra

Image_to_VideoLTX = tk.Button(buttons_frame, text="FramePack ImagetoVideo_F1",
                           command=FramePack_ImageToVideo_F1, bg='#ffd076', fg='black')
Image_to_VideoLTX.pack(side='left')

def setup_drag_drop(window):
    # Assicurati che window sia un'istanza di TkinterDnD.Tk
    if not isinstance(window, TkinterDnD.Tk):
        print("ERRORE: La finestra deve essere un'istanza di TkinterDnD.Tk")
        print("Usa 'window = TkinterDnD.Tk()' invece di 'window = tk.Tk()'")
        return
    
    # Configura gli eventi per il drag and drop su entrambi i canvas
    try:
        # Registra i canvas come target di drop
        canvas.drop_target_register(DND_FILES)
        canvas.dnd_bind('<<Drop>>', lambda event: on_drop(event, 1))
        canvas.bind("<Button-1>", lambda event: on_canvas_click(event, 1))
        
        canvas2.drop_target_register(DND_FILES)
        canvas2.dnd_bind('<<Drop>>', lambda event: on_drop(event, 2))
        canvas2.bind("<Button-1>", lambda event: on_canvas_click(event, 2))
        
        print("✅ Drag and drop configurato correttamente")
    except Exception as e:
        print(f"❌ Errore nella configurazione drag and drop: {e}")
        print("Assicurati di aver installato tkinterdnd2 con: pip install tkinterdnd2")

def on_drop(event, canvas_num):
    # Gestisce il drop di un'immagine sul canvas specificato
    file_path = event.data
    
    # Pulizia del percorso file (gestisce Windows, Linux e macOS)
    if file_path.startswith('{') and file_path.endswith('}'):
        file_path = file_path[1:-1]
    if file_path.startswith('file://'):
        file_path = file_path[7:]
    # Rimuovi eventuali caratteri di escape in Windows
    file_path = file_path.replace('\\', '/')
    
    print(f"File ricevuto: {file_path}")
    
    # Verifica che il file esista e sia un'immagine valida
    if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        process_image(file_path, canvas_num)
    else:
        print(f"File non valido o non esistente: {file_path}")

def on_canvas_click(event, canvas_num):
    # Apre il selettore di file quando si clicca sul canvas
    load_image(canvas_num)

def load_image(canvas_num):
    # Apre il selettore di file
    file_path = filedialog.askopenfilename(
        title=f"Seleziona un'immagine per il canvas {canvas_num}",
        filetypes=[("Immagini", "*.png *.jpg *.jpeg *.gif *.bmp")]
    )
    
    if file_path:
        process_image(file_path, canvas_num)

def process_image(file_path, canvas_num):
    global current_image1, photo_image1, current_image2, photo_image2, ip_image_path1, ip_image_path2
    
    try:
        # Carica l'immagine con PIL
        img = Image.open(file_path)
        
        # Determina quale canvas e variabili usare
        if canvas_num == 1:
            canvas_obj = canvas
            frame_obj = frame_canvas1
            canvas_text_obj = canvas_text
            current_image_var = "current_image1"
            photo_image_var = "photo_image1"
            ip_path_var = "ip_image_path1"
            canvas_label = "Immagine Faccia"
        else:
            canvas_obj = canvas2
            frame_obj = frame_canvas2
            canvas_text_obj = canvas_text2
            current_image_var = "current_image2"
            photo_image_var = "photo_image2"
            ip_path_var = "ip_image_path2"
            canvas_label = "Immagine Intera"
        
        # Ottieni le dimensioni del canvas
        canvas_width = frame_obj.winfo_width()
        canvas_height = frame_obj.winfo_height()
        
        if canvas_width <= 1:  # Se il frame non è ancora stato renderizzato completamente
            canvas_width = 250
        if canvas_height <= 1:
            canvas_height = 150
        
        # Ridimensiona l'immagine mantenendo le proporzioni
        img = resize_image(img, canvas_width, canvas_height)
        
        # Converti l'immagine in formato compatibile con tkinter
        photo_img = ImageTk.PhotoImage(img)
        
        # Rimuovi il testo
        canvas_obj.delete(canvas_text_obj)
        
        # Rimuovi l'immagine precedente se esiste
        current_img = globals().get(current_image_var)
        if current_img:
            canvas_obj.delete(current_img)
        
        # Visualizza la nuova immagine
        new_image = canvas_obj.create_image(
            canvas_width // 2, canvas_height // 2,
            image=photo_img, anchor='center'
        )
        
        # Aggiorna le variabili globali
        globals()[current_image_var] = new_image
        globals()[photo_image_var] = photo_img
        globals()[ip_path_var] = file_path
        
        print(f"✅ Immagine IP adapter {canvas_num} caricata: {file_path}")
        
    except Exception as e:
        print(f"❌ Errore nel caricamento dell'immagine: {e}")
        # Ripristina il testo se c'è un errore
        if canvas_num == 1:
            globals()["canvas_text"] = canvas.create_text(125, 75, text=f"IP Adapter:\n{canvas_label}", 
                                        font=("Arial", 14), fill='white')
        else:
            globals()["canvas_text2"] = canvas2.create_text(125, 75, text=f"IP Adapter:\n{canvas_label}", 
                                         font=("Arial", 14), fill='white')

def resize_image(img, width, height):
    # Ridimensiona l'immagine mantenendo le proporzioni
    img_width, img_height = img.size
    
    # Calcola il rapporto per il ridimensionamento
    ratio = min(width / img_width, height / img_height)
    new_width = int(img_width * ratio)
    new_height = int(img_height * ratio)
    
    # Ridimensiona l'immagine
    return img.resize((new_width, new_height), Image.LANCZOS)

# Da chiamare all'inizializzazione dell'applicazione
def init_drag_drop(main_window):
    # Inizializza il drag and drop
    try:
        setup_drag_drop(main_window)
    except Exception as e:
        print(f"❌ Errore nell'inizializzazione del drag and drop: {e}")
        print("Assicurati di aver importato tkinterdnd2 e creato la finestra con TkinterDnD.Tk()")


# --- CREAZIONE CARTELLE NECESSARIE ---
for d in output_dirs:
    os.makedirs(d, exist_ok=True)
init_drag_drop(window)

# --- AVVIO ---
status_var.set("Applicazione avviata. Pronta all'uso.")
window.mainloop()